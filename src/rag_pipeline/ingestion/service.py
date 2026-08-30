"""Ingestion orchestration: source path -> persisted normalized document.

    source path
    -> read/hash original bytes
    -> select loader
    -> normalize/extract
    -> construct canonical document
    -> persist raw source
    -> persist processed representation
    -> return canonical normalized document

This intentionally stops at a normalized, persisted document. It does not
chunk, embed, or index anything.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from ..config import Settings
from .exceptions import SourceNotFoundError
from .loader import load_segments
from .models import NormalizedDocument
from .storage import write_processed, write_raw

logger = logging.getLogger(__name__)


def compute_document_id(data: bytes) -> str:
    """Deterministic content identity: full SHA-256 hex digest of the raw bytes."""
    return hashlib.sha256(data).hexdigest()


def ingest_document(
    source_path: Path | str, settings: Settings | None = None
) -> NormalizedDocument:
    """Ingest a single local source file into a persisted normalized document."""
    path = Path(source_path)
    if not path.is_file():
        raise SourceNotFoundError(f"Source file not found: {path}")

    settings = settings or Settings()
    raw_bytes = path.read_bytes()
    document_id = compute_document_id(raw_bytes)

    logger.info("ingestion started file=%s document_id=%s", path.name, document_id)

    segments, file_type = load_segments(path, raw_bytes)

    logger.info(
        "detected format=%s document_id=%s segment_count=%d",
        file_type,
        document_id,
        len(segments),
    )

    raw_target = write_raw(settings.raw_data_dir, document_id, path.name, raw_bytes)
    raw_path = raw_target.relative_to(settings.raw_data_dir).as_posix()

    document = NormalizedDocument(
        document_id=document_id,
        source_file=path.name,
        file_type=file_type,
        raw_path=raw_path,
        segments=tuple(segments),
    )

    write_processed(settings.processed_data_dir, document)

    logger.info("persistence completed document_id=%s", document_id)

    return document
