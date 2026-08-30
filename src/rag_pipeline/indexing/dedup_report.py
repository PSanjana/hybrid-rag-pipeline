"""Schema-versioned persistence for one deduplication run's duplicate report.

Mirrors `sparse.py`'s persistence pattern: one JSON file per snapshot,
written atomically (temp file + rename), portable (no pickle). This module
is the only part of the indexing pipeline that knows how a
`DeduplicationResult` gets written to and read from disk --
`rag_pipeline.deduplication` itself has no filesystem or `Settings`
dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Settings
from ..deduplication.models import DeduplicationResult, DuplicateRecord
from .exceptions import DedupReportError

DEDUP_REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class DuplicateReport:
    """The portable, persisted form of one deduplication run's audit trail."""

    schema_version: int
    snapshot_id: str
    dedup_algorithm_version: str
    dedup_similarity_threshold: float
    duplicates: tuple[DuplicateRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "dedup_algorithm_version": self.dedup_algorithm_version,
            "dedup_similarity_threshold": self.dedup_similarity_threshold,
            "duplicates": [record.to_dict() for record in self.duplicates],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DuplicateReport:
        version = data.get("schema_version")
        if version != DEDUP_REPORT_SCHEMA_VERSION:
            raise DedupReportError(
                f"Unsupported duplicate report schema_version={version!r}; expected "
                f"{DEDUP_REPORT_SCHEMA_VERSION}."
            )
        return cls(
            schema_version=version,
            snapshot_id=data["snapshot_id"],
            dedup_algorithm_version=data["dedup_algorithm_version"],
            dedup_similarity_threshold=data["dedup_similarity_threshold"],
            duplicates=tuple(DuplicateRecord.from_dict(item) for item in data["duplicates"]),
        )


def dedup_report_dir(settings: Settings, snapshot_id: str) -> Path:
    return settings.index_root_dir / "dedup" / snapshot_id


def dedup_report_path(settings: Settings, snapshot_id: str) -> Path:
    return dedup_report_dir(settings, snapshot_id) / "duplicates.json"


def write_dedup_report(settings: Settings, snapshot_id: str, result: DeduplicationResult) -> Path:
    """Persist `result`'s duplicate records for `snapshot_id`, atomically.

    An empty `result.duplicates` is valid and persists as an empty list --
    "zero duplicates found" is a normal, auditable outcome, not an error.
    """
    report = DuplicateReport(
        schema_version=DEDUP_REPORT_SCHEMA_VERSION,
        snapshot_id=snapshot_id,
        dedup_algorithm_version=result.algorithm_version,
        dedup_similarity_threshold=result.similarity_threshold,
        duplicates=result.duplicates,
    )
    path = dedup_report_path(settings, snapshot_id)
    payload = json.dumps(report.to_dict(), ensure_ascii=False)
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(path)
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise DedupReportError(f"Failed to write duplicate report to {path}: {exc}") from exc
    return path


def load_dedup_report(settings: Settings, snapshot_id: str) -> DuplicateReport:
    """Load and validate a persisted duplicate report."""
    path = dedup_report_path(settings, snapshot_id)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise DedupReportError(f"No duplicate report found for snapshot {snapshot_id!r}.") from exc
    except OSError as exc:
        raise DedupReportError(f"Failed to read duplicate report {path}: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DedupReportError(f"Duplicate report {path} is corrupt: {exc}") from exc

    report = DuplicateReport.from_dict(data)
    if report.snapshot_id != snapshot_id:
        raise DedupReportError(
            f"Duplicate report at {path} has snapshot_id={report.snapshot_id!r}, "
            f"expected {snapshot_id!r}."
        )
    return report
