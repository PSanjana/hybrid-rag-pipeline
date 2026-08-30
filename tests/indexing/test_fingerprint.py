"""Tests for rag_pipeline.indexing.fingerprint."""

from rag_pipeline.config import ChunkingStrategy
from rag_pipeline.indexing.fingerprint import compute_snapshot_id

_IDS = ("id1", "id2", "id3")


def test_identical_inputs_produce_identical_snapshot_id() -> None:
    first = compute_snapshot_id(
        _IDS, ChunkingStrategy.RECURSIVE, "text-embedding-3-small", "technical_v1"
    )
    second = compute_snapshot_id(
        _IDS, ChunkingStrategy.RECURSIVE, "text-embedding-3-small", "technical_v1"
    )
    assert first == second


def test_snapshot_id_is_sha256_hex() -> None:
    snapshot_id = compute_snapshot_id(
        _IDS, ChunkingStrategy.RECURSIVE, "text-embedding-3-small", "technical_v1"
    )
    assert len(snapshot_id) == 64
    int(snapshot_id, 16)  # raises ValueError if not valid hex


def test_changed_chunk_id_changes_snapshot_id() -> None:
    baseline = compute_snapshot_id(
        _IDS, ChunkingStrategy.RECURSIVE, "text-embedding-3-small", "technical_v1"
    )
    changed = compute_snapshot_id(
        ("id1", "id2", "DIFFERENT"),
        ChunkingStrategy.RECURSIVE,
        "text-embedding-3-small",
        "technical_v1",
    )
    assert baseline != changed


def test_changed_chunk_order_changes_snapshot_id() -> None:
    baseline = compute_snapshot_id(
        _IDS, ChunkingStrategy.RECURSIVE, "text-embedding-3-small", "technical_v1"
    )
    reordered = compute_snapshot_id(
        ("id3", "id2", "id1"), ChunkingStrategy.RECURSIVE, "text-embedding-3-small", "technical_v1"
    )
    assert baseline != reordered


def test_changed_embedding_model_changes_snapshot_id() -> None:
    baseline = compute_snapshot_id(
        _IDS, ChunkingStrategy.RECURSIVE, "text-embedding-3-small", "technical_v1"
    )
    changed = compute_snapshot_id(
        _IDS, ChunkingStrategy.RECURSIVE, "text-embedding-3-large", "technical_v1"
    )
    assert baseline != changed


def test_changed_tokenizer_version_changes_snapshot_id() -> None:
    baseline = compute_snapshot_id(
        _IDS, ChunkingStrategy.RECURSIVE, "text-embedding-3-small", "technical_v1"
    )
    changed = compute_snapshot_id(
        _IDS, ChunkingStrategy.RECURSIVE, "text-embedding-3-small", "technical_v2"
    )
    assert baseline != changed


def test_different_chunking_strategy_changes_snapshot_id() -> None:
    recursive = compute_snapshot_id(
        _IDS, ChunkingStrategy.RECURSIVE, "text-embedding-3-small", "technical_v1"
    )
    fixed = compute_snapshot_id(
        _IDS, ChunkingStrategy.FIXED, "text-embedding-3-small", "technical_v1"
    )
    semantic = compute_snapshot_id(
        _IDS, ChunkingStrategy.SEMANTIC, "text-embedding-3-small", "technical_v1"
    )
    assert len({recursive, fixed, semantic}) == 3
