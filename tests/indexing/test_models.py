"""Tests for rag_pipeline.indexing.models (canonical_order, IndexManifest)."""

import pytest

from rag_pipeline.chunking.models import Chunk, build_chunk
from rag_pipeline.config import ChunkingStrategy
from rag_pipeline.indexing.exceptions import InvalidChunkCorpusError
from rag_pipeline.indexing.models import MANIFEST_SCHEMA_VERSION, IndexManifest, canonical_order

from .conftest import make_chunks


def _chunk(document_id: str, chunk_index: int, text: str = "content") -> Chunk:
    return build_chunk(
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
        source_file="doc.md",
        section_heading=None,
        page_number=None,
        strategy=ChunkingStrategy.RECURSIVE,
    )


def test_empty_corpus_is_rejected() -> None:
    with pytest.raises(InvalidChunkCorpusError):
        canonical_order([])


def test_canonical_order_is_independent_of_input_order() -> None:
    a = _chunk("doc-a" * 12 + "aaaa", 0)
    b = _chunk("doc-a" * 12 + "aaaa", 1)
    c = _chunk("doc-b" * 12 + "bbbb", 0)

    forward = canonical_order([a, b, c])
    shuffled = canonical_order([c, b, a])
    reversed_order = canonical_order([c, a, b])

    assert forward == shuffled == reversed_order


def test_canonical_order_sorts_by_document_then_chunk_index() -> None:
    doc_a = "a" * 64
    doc_b = "b" * 64
    chunks = [
        _chunk(doc_b, 0, "b0"),
        _chunk(doc_a, 1, "a1"),
        _chunk(doc_a, 0, "a0"),
    ]
    ordered = canonical_order(chunks)
    assert [c.document_id for c in ordered] == [doc_a, doc_a, doc_b]
    assert [c.chunk_index for c in ordered] == [0, 1, 0]


def test_canonical_order_does_not_mutate_input() -> None:
    chunks = make_chunks(3)
    original = list(chunks)
    canonical_order(chunks)
    assert chunks == original


def test_duplicate_chunk_id_is_rejected() -> None:
    chunk = _chunk("d" * 64, 0)
    with pytest.raises(InvalidChunkCorpusError, match="Duplicate"):
        canonical_order([chunk, chunk])


def test_duplicate_document_chunk_index_position_with_different_chunk_ids_is_rejected() -> None:
    doc_id = "d" * 64
    # Different text -> different chunk_id, but the same (document_id,
    # chunk_index) position -- e.g. two different chunking runs' outputs
    # merged together incorrectly.
    first = _chunk(doc_id, 0, text="first version of this chunk")
    second = _chunk(doc_id, 0, text="second, different version of this chunk")
    assert first.chunk_id != second.chunk_id

    with pytest.raises(InvalidChunkCorpusError, match="position"):
        canonical_order([first, second])


def test_identical_chunk_index_in_different_documents_remains_valid() -> None:
    doc_a = "a" * 64
    doc_b = "b" * 64
    chunk_a = _chunk(doc_a, 0, text="content from document a")
    chunk_b = _chunk(doc_b, 0, text="content from document b")

    ordered = canonical_order([chunk_a, chunk_b])

    assert len(ordered) == 2
    assert {chunk.document_id for chunk in ordered} == {doc_a, doc_b}
    assert all(chunk.chunk_index == 0 for chunk in ordered)


def test_empty_text_is_rejected() -> None:
    empty = _chunk("d" * 64, 0, text="   ")
    with pytest.raises(InvalidChunkCorpusError, match="empty"):
        canonical_order([empty])


def test_mixed_chunking_strategies_are_rejected() -> None:
    doc_id = "d" * 64
    recursive_chunk = build_chunk(
        document_id=doc_id,
        chunk_index=0,
        text="recursive content",
        source_file="doc.md",
        section_heading=None,
        page_number=None,
        strategy=ChunkingStrategy.RECURSIVE,
    )
    fixed_chunk = build_chunk(
        document_id=doc_id,
        chunk_index=1,
        text="fixed content",
        source_file="doc.md",
        section_heading=None,
        page_number=None,
        strategy=ChunkingStrategy.FIXED,
    )
    with pytest.raises(InvalidChunkCorpusError, match="strategy"):
        canonical_order([recursive_chunk, fixed_chunk])


def test_index_manifest_round_trips_through_dict() -> None:
    manifest = IndexManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        snapshot_id="a" * 64,
        request_fingerprint="b" * 64,
        chunking_strategy=ChunkingStrategy.RECURSIVE,
        embedding_model="text-embedding-3-small",
        embedding_dimension=8,
        bm25_tokenizer_version="technical_v1",
        dedup_algorithm_version="cosine_v1",
        dedup_similarity_threshold=0.95,
        pre_dedup_chunk_count=3,
        chunk_count=2,
        duplicate_count=1,
        chunk_ids=("id1", "id2"),
        chroma_collection_name="rag-recursive-abc123",
        sparse_snapshot_path="/tmp/corpus.json",
        dedup_report_path="/tmp/duplicates.json",
        created_at="2026-01-01T00:00:00+00:00",
    )
    reloaded = IndexManifest.from_dict(manifest.to_dict())
    assert reloaded == manifest
