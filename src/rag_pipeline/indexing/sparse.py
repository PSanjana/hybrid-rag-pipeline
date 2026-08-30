"""BM25 sparse index: corpus persistence, and reconstruction after a process restart.

`rank_bm25.BM25Okapi` is never pickled — pickle ties a snapshot to a
specific library/Python version and is unsafe to load from untrusted
sources. Instead, a small, portable, schema-versioned corpus snapshot
(chunk IDs + texts, in canonical order) is persisted as JSON, and BM25 is
rebuilt from it (via the shared tokenizer) whenever it's needed.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from ..chunking.models import Chunk
from ..config import Settings
from .exceptions import SparseIndexError
from .tokenizer import TOKENIZER_VERSION, tokenize

SPARSE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SparseCorpusSnapshot:
    """The portable, persisted form of a BM25 corpus (not the BM25Okapi object itself)."""

    schema_version: int
    snapshot_id: str
    tokenizer_version: str
    chunk_ids: tuple[str, ...]
    texts: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "tokenizer_version": self.tokenizer_version,
            "chunk_ids": list(self.chunk_ids),
            "texts": list(self.texts),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SparseCorpusSnapshot:
        version = data.get("schema_version")
        if version != SPARSE_SCHEMA_VERSION:
            raise SparseIndexError(
                f"Unsupported sparse snapshot schema_version={version!r}; expected "
                f"{SPARSE_SCHEMA_VERSION}."
            )
        return cls(
            schema_version=version,
            snapshot_id=data["snapshot_id"],
            tokenizer_version=data["tokenizer_version"],
            chunk_ids=tuple(data["chunk_ids"]),
            texts=tuple(data["texts"]),
        )


@dataclass(frozen=True, slots=True)
class ReconstructedBM25Index:
    """A rebuilt BM25 index plus its exact corpus-position -> chunk_id mapping."""

    bm25: BM25Okapi
    chunk_ids: tuple[str, ...]


def sparse_snapshot_dir(settings: Settings, snapshot_id: str) -> Path:
    return settings.bm25_dir / snapshot_id


def sparse_corpus_path(settings: Settings, snapshot_id: str) -> Path:
    return sparse_snapshot_dir(settings, snapshot_id) / "corpus.json"


def write_sparse_snapshot(
    settings: Settings, chunks: Sequence[Chunk], snapshot_id: str, tokenizer_version: str
) -> Path:
    """Persist the canonically-ordered corpus for `snapshot_id`, atomically.

    Writes to a temporary file and renames it into place, so an interrupted
    write never leaves a partial `corpus.json` mistaken for a valid one.
    """
    snapshot = SparseCorpusSnapshot(
        schema_version=SPARSE_SCHEMA_VERSION,
        snapshot_id=snapshot_id,
        tokenizer_version=tokenizer_version,
        chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
        texts=tuple(chunk.text for chunk in chunks),
    )
    path = sparse_corpus_path(settings, snapshot_id)
    payload = json.dumps(snapshot.to_dict(), ensure_ascii=False)
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(path)
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise SparseIndexError(f"Failed to write sparse snapshot to {path}: {exc}") from exc
    return path


def load_sparse_snapshot(
    settings: Settings,
    snapshot_id: str,
    expected_tokenizer_version: str = TOKENIZER_VERSION,
) -> SparseCorpusSnapshot:
    """Load and validate a persisted sparse snapshot.

    Rejects a snapshot built with a different `tokenizer_version` than
    `expected_tokenizer_version` (the current tokenizer by default): the
    persisted texts would be re-tokenized under different semantics than
    the ones the snapshot was built and fingerprinted with, silently
    corrupting the sparse index rather than raising.
    """
    path = sparse_corpus_path(settings, snapshot_id)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SparseIndexError(f"No sparse snapshot found for id {snapshot_id!r}.") from exc
    except OSError as exc:
        raise SparseIndexError(f"Failed to read sparse snapshot {path}: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SparseIndexError(f"Sparse snapshot {path} is corrupt: {exc}") from exc

    snapshot = SparseCorpusSnapshot.from_dict(data)
    if snapshot.snapshot_id != snapshot_id:
        raise SparseIndexError(
            f"Sparse snapshot at {path} has snapshot_id={snapshot.snapshot_id!r}, "
            f"expected {snapshot_id!r}."
        )
    if snapshot.tokenizer_version != expected_tokenizer_version:
        raise SparseIndexError(
            f"Sparse snapshot {path} was built with tokenizer_version="
            f"{snapshot.tokenizer_version!r}, but the expected/current tokenizer is "
            f"{expected_tokenizer_version!r}."
        )
    if len(snapshot.chunk_ids) != len(snapshot.texts):
        raise SparseIndexError(
            f"Sparse snapshot {path} is malformed: {len(snapshot.chunk_ids)} chunk_ids "
            f"but {len(snapshot.texts)} texts."
        )
    if len(set(snapshot.chunk_ids)) != len(snapshot.chunk_ids):
        raise SparseIndexError(
            f"Sparse snapshot {path} is malformed: contains duplicate chunk_ids. Each "
            "chunk_id must appear exactly once."
        )
    return snapshot


def build_bm25_index(chunks: Sequence[Chunk]) -> ReconstructedBM25Index:
    """Build a BM25 index directly from canonically-ordered chunks (at index-build time)."""
    tokenized_corpus = [tokenize(chunk.text) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return ReconstructedBM25Index(bm25=bm25, chunk_ids=tuple(chunk.chunk_id for chunk in chunks))


def load_bm25_index(
    settings: Settings,
    snapshot_id: str,
    expected_tokenizer_version: str = TOKENIZER_VERSION,
) -> ReconstructedBM25Index:
    """Reload a persisted sparse snapshot and reconstruct its `BM25Okapi` index."""
    snapshot = load_sparse_snapshot(settings, snapshot_id, expected_tokenizer_version)
    tokenized_corpus = [tokenize(text) for text in snapshot.texts]
    bm25 = BM25Okapi(tokenized_corpus)
    return ReconstructedBM25Index(bm25=bm25, chunk_ids=snapshot.chunk_ids)
