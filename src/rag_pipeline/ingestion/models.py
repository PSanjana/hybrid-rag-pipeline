"""Canonical, format-independent representation of an ingested document.

A `Segment` is a normalized section or page extracted from a source file.
It is intentionally *not* a retrieval chunk: no size-based splitting,
overlap, or token-budgeting happens here. Chunking is a later pipeline step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .exceptions import PersistenceError

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class Segment:
    """A single normalized content segment (e.g. one Markdown section or one PDF page)."""

    text: str
    section_heading: str | None = None
    page_number: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "section_heading": self.section_heading,
            "page_number": self.page_number,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Segment:
        return cls(
            text=data["text"],
            section_heading=data.get("section_heading"),
            page_number=data.get("page_number"),
        )


@dataclass(frozen=True, slots=True)
class NormalizedDocument:
    """Document-level metadata plus its ordered content segments."""

    document_id: str
    source_file: str
    file_type: str
    raw_path: str
    segments: tuple[Segment, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "document_id": self.document_id,
            "source_file": self.source_file,
            "file_type": self.file_type,
            "raw_path": self.raw_path,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizedDocument:
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise PersistenceError(
                f"Unsupported processed document schema_version={version!r}; "
                f"expected {SCHEMA_VERSION}."
            )
        return cls(
            document_id=data["document_id"],
            source_file=data["source_file"],
            file_type=data["file_type"],
            raw_path=data["raw_path"],
            segments=tuple(Segment.from_dict(item) for item in data["segments"]),
        )
