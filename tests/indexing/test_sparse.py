"""Tests for rag_pipeline.indexing.sparse (BM25 corpus persistence/reconstruction)."""

import json

import pytest
from rank_bm25 import BM25Okapi

from rag_pipeline.chunking.models import build_chunk
from rag_pipeline.config import ChunkingStrategy, Settings
from rag_pipeline.indexing.exceptions import SparseIndexError
from rag_pipeline.indexing.sparse import (
    build_bm25_index,
    load_bm25_index,
    load_sparse_snapshot,
    sparse_corpus_path,
    write_sparse_snapshot,
)
from rag_pipeline.indexing.tokenizer import TOKENIZER_VERSION

from .conftest import make_chunks


def test_sparse_snapshot_is_persisted(index_settings: Settings) -> None:
    chunks = make_chunks(3)
    path = write_sparse_snapshot(index_settings, chunks, "snap-a", TOKENIZER_VERSION)
    assert path.exists()
    assert path == sparse_corpus_path(index_settings, "snap-a")


def test_persisted_chunk_id_order_equals_canonical_order(index_settings: Settings) -> None:
    chunks = make_chunks(5)
    write_sparse_snapshot(index_settings, chunks, "snap-b", TOKENIZER_VERSION)
    snapshot = load_sparse_snapshot(index_settings, "snap-b")
    assert snapshot.chunk_ids == tuple(chunk.chunk_id for chunk in chunks)


def test_bm25_can_be_reconstructed_after_reload(index_settings: Settings) -> None:
    chunks = make_chunks(4)
    write_sparse_snapshot(index_settings, chunks, "snap-c", TOKENIZER_VERSION)

    reconstructed = load_bm25_index(index_settings, "snap-c")

    assert isinstance(reconstructed.bm25, BM25Okapi)


def test_reconstructed_corpus_length_equals_chunk_count(index_settings: Settings) -> None:
    chunks = make_chunks(6)
    write_sparse_snapshot(index_settings, chunks, "snap-d", TOKENIZER_VERSION)

    reconstructed = load_bm25_index(index_settings, "snap-d")

    assert reconstructed.bm25.corpus_size == 6
    assert len(reconstructed.chunk_ids) == 6


def test_bm25_position_to_chunk_id_mapping_is_exact(index_settings: Settings) -> None:
    chunks = make_chunks(5)
    write_sparse_snapshot(index_settings, chunks, "snap-e", TOKENIZER_VERSION)

    reconstructed = load_bm25_index(index_settings, "snap-e")

    assert reconstructed.chunk_ids == tuple(chunk.chunk_id for chunk in chunks)


def test_no_pickle_is_used_in_the_persisted_snapshot(index_settings: Settings) -> None:
    chunks = make_chunks(2)
    path = write_sparse_snapshot(index_settings, chunks, "snap-f", TOKENIZER_VERSION)
    # The persisted artifact is plain JSON, not a pickle stream.
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert data["schema_version"] == 1


def test_build_bm25_index_directly_from_chunks_matches_reload(index_settings: Settings) -> None:
    chunks = make_chunks(3)
    direct = build_bm25_index(chunks)
    write_sparse_snapshot(index_settings, chunks, "snap-g", TOKENIZER_VERSION)
    reloaded = load_bm25_index(index_settings, "snap-g")

    assert direct.chunk_ids == reloaded.chunk_ids
    assert direct.bm25.corpus_size == reloaded.bm25.corpus_size
    assert direct.bm25.doc_len == reloaded.bm25.doc_len


def test_identifier_survives_and_scores_for_a_matching_query(index_settings: Settings) -> None:
    chunks = [
        build_chunk(
            document_id="d" * 64,
            chunk_index=0,
            text="ERR_DB_1042 means the connection pool timed out.",
            source_file="errors.md",
            section_heading=None,
            page_number=None,
            strategy=ChunkingStrategy.RECURSIVE,
        ),
        build_chunk(
            document_id="d" * 64,
            chunk_index=1,
            text="Rate limiting returns ERR_RATE_4290 when exceeded.",
            source_file="errors.md",
            section_heading=None,
            page_number=None,
            strategy=ChunkingStrategy.RECURSIVE,
        ),
        build_chunk(
            document_id="d" * 64,
            chunk_index=2,
            text="Unrelated content about onboarding new employees.",
            source_file="handbook.md",
            section_heading=None,
            page_number=None,
            strategy=ChunkingStrategy.RECURSIVE,
        ),
    ]
    reconstructed = build_bm25_index(chunks)

    from rag_pipeline.indexing.tokenizer import tokenize

    scores = reconstructed.bm25.get_scores(tokenize("ERR_DB_1042"))

    # The chunk that actually contains the identifier must score highest.
    top_index = max(range(len(scores)), key=lambda i: scores[i])
    assert reconstructed.chunk_ids[top_index] == chunks[0].chunk_id
    assert scores[0] > scores[2]


def test_load_missing_snapshot_raises(index_settings: Settings) -> None:
    with pytest.raises(SparseIndexError):
        load_sparse_snapshot(index_settings, "does-not-exist")


def test_load_corrupt_snapshot_raises(index_settings: Settings) -> None:
    path = sparse_corpus_path(index_settings, "snap-corrupt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not valid json", encoding="utf-8")
    with pytest.raises(SparseIndexError):
        load_sparse_snapshot(index_settings, "snap-corrupt")


def test_snapshot_id_mismatch_is_rejected(index_settings: Settings) -> None:
    chunks = make_chunks(2)
    write_sparse_snapshot(index_settings, chunks, "snap-h", TOKENIZER_VERSION)
    # Simulate reading it back under a different expected snapshot_id.
    with pytest.raises(SparseIndexError):
        load_sparse_snapshot(index_settings, "snap-h-typo")


def test_tokenizer_version_mismatch_is_rejected_by_default(index_settings: Settings) -> None:
    chunks = make_chunks(2)
    write_sparse_snapshot(index_settings, chunks, "snap-i", "some_old_tokenizer_v0")
    with pytest.raises(SparseIndexError, match="tokenizer_version"):
        load_sparse_snapshot(index_settings, "snap-i")


def test_tokenizer_version_mismatch_rejected_with_explicit_expected_version(
    index_settings: Settings,
) -> None:
    chunks = make_chunks(2)
    write_sparse_snapshot(index_settings, chunks, "snap-j", TOKENIZER_VERSION)
    with pytest.raises(SparseIndexError, match="tokenizer_version"):
        load_sparse_snapshot(index_settings, "snap-j", expected_tokenizer_version="technical_v2")


def test_matching_tokenizer_version_is_accepted(index_settings: Settings) -> None:
    chunks = make_chunks(2)
    write_sparse_snapshot(index_settings, chunks, "snap-k", TOKENIZER_VERSION)
    snapshot = load_sparse_snapshot(
        index_settings, "snap-k", expected_tokenizer_version=TOKENIZER_VERSION
    )
    assert snapshot.tokenizer_version == TOKENIZER_VERSION


def test_load_bm25_index_rejects_tokenizer_version_mismatch(index_settings: Settings) -> None:
    chunks = make_chunks(2)
    write_sparse_snapshot(index_settings, chunks, "snap-l", "some_old_tokenizer_v0")
    with pytest.raises(SparseIndexError, match="tokenizer_version"):
        load_bm25_index(index_settings, "snap-l")


def test_duplicate_chunk_ids_in_persisted_snapshot_are_rejected(index_settings: Settings) -> None:
    chunks = make_chunks(3)
    write_sparse_snapshot(index_settings, chunks, "snap-dup", TOKENIZER_VERSION)

    path = sparse_corpus_path(index_settings, "snap-dup")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["chunk_ids"][1] = data["chunk_ids"][0]  # same count, one id duplicated
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(SparseIndexError, match="duplicate"):
        load_sparse_snapshot(index_settings, "snap-dup")


def test_load_bm25_index_rejects_duplicate_chunk_ids(index_settings: Settings) -> None:
    chunks = make_chunks(3)
    write_sparse_snapshot(index_settings, chunks, "snap-dup2", TOKENIZER_VERSION)

    path = sparse_corpus_path(index_settings, "snap-dup2")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["chunk_ids"][1] = data["chunk_ids"][0]
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(SparseIndexError, match="duplicate"):
        load_bm25_index(index_settings, "snap-dup2")
