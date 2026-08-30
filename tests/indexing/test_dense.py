"""Tests for rag_pipeline.indexing.dense (Chroma dense index)."""

import chromadb
import pytest

from rag_pipeline.chunking.models import build_chunk
from rag_pipeline.config import ChunkingStrategy, Settings
from rag_pipeline.indexing.dense import (
    build_collection_name,
    build_dense_index,
    get_chroma_client,
    get_dense_collection_ids,
    verify_dense_collection,
)
from rag_pipeline.indexing.exceptions import DenseIndexError

from .conftest import make_chunks


def test_build_collection_name_pattern() -> None:
    name = build_collection_name(ChunkingStrategy.RECURSIVE, "a" * 64, prefix="rag")
    assert name == f"rag-recursive-{'a' * 12}"


def test_local_persistent_collection_is_created(index_settings: Settings) -> None:
    client = get_chroma_client(index_settings)
    chunks = make_chunks(3)
    embeddings = [[0.1, 0.2, 0.3] for _ in chunks]

    collection = build_dense_index(
        client, "rag-recursive-abc123def456", chunks, embeddings, batch_size=10
    )

    assert collection.name == "rag-recursive-abc123def456"
    assert index_settings.chroma_dir.exists()


def test_collection_uses_cosine_space(index_settings: Settings) -> None:
    client = get_chroma_client(index_settings)
    chunks = make_chunks(2)
    embeddings = [[1.0, 0.0], [0.0, 1.0]]

    collection = build_dense_index(
        client, "rag-recursive-cosine00001", chunks, embeddings, batch_size=10
    )

    assert collection.configuration_json["hnsw"]["space"] == "cosine"


def test_one_record_exists_per_chunk(index_settings: Settings) -> None:
    client = get_chroma_client(index_settings)
    chunks = make_chunks(5)
    embeddings = [[float(i), 0.0] for i in range(5)]

    collection = build_dense_index(
        client, "rag-recursive-count000001", chunks, embeddings, batch_size=2
    )

    assert collection.count() == 5


def test_record_ids_exactly_equal_chunk_ids(index_settings: Settings) -> None:
    client = get_chroma_client(index_settings)
    chunks = make_chunks(4)
    embeddings = [[float(i), 0.0] for i in range(4)]

    collection = build_dense_index(
        client, "rag-recursive-idmatch00001", chunks, embeddings, batch_size=10
    )

    stored_ids = set(get_dense_collection_ids(collection))
    expected_ids = {chunk.chunk_id for chunk in chunks}
    assert stored_ids == expected_ids


def test_stored_documents_equal_chunk_text(index_settings: Settings) -> None:
    client = get_chroma_client(index_settings)
    chunks = make_chunks(2)
    embeddings = [[0.1, 0.2], [0.3, 0.4]]

    collection = build_dense_index(
        client, "rag-recursive-doctext000001", chunks, embeddings, batch_size=10
    )

    result = collection.get(ids=[chunks[0].chunk_id], include=["documents"])
    assert result["documents"] == [chunks[0].text]


def test_required_metadata_is_present(index_settings: Settings) -> None:
    client = get_chroma_client(index_settings)
    chunk = build_chunk(
        document_id="d" * 64,
        chunk_index=0,
        text="Some content",
        source_file="doc.md",
        section_heading="Intro > Setup",
        page_number=3,
        strategy=ChunkingStrategy.RECURSIVE,
    )
    collection = build_dense_index(
        client, "rag-recursive-metadata00001", [chunk], [[1.0, 2.0]], batch_size=10
    )

    result = collection.get(ids=[chunk.chunk_id], include=["metadatas"])
    metadata = result["metadatas"][0]
    assert metadata["document_id"] == chunk.document_id
    assert metadata["chunk_index"] == chunk.chunk_index
    assert metadata["source_file"] == chunk.source_file
    assert metadata["chunking_strategy"] == chunk.chunking_strategy.value
    assert metadata["character_count"] == chunk.character_count
    assert metadata["section_heading"] == "Intro > Setup"
    assert metadata["page_number"] == 3


def test_none_page_and_heading_metadata_are_omitted_not_null(index_settings: Settings) -> None:
    client = get_chroma_client(index_settings)
    chunk = build_chunk(
        document_id="d" * 64,
        chunk_index=0,
        text="Some content",
        source_file="doc.txt",
        section_heading=None,
        page_number=None,
        strategy=ChunkingStrategy.RECURSIVE,
    )
    collection = build_dense_index(
        client, "rag-recursive-nonemeta00001", [chunk], [[1.0, 2.0]], batch_size=10
    )

    result = collection.get(ids=[chunk.chunk_id], include=["metadatas"])
    metadata = result["metadatas"][0]
    assert "section_heading" not in metadata
    assert "page_number" not in metadata


def test_correct_embeddings_associated_with_correct_chunk_ids(index_settings: Settings) -> None:
    client = get_chroma_client(index_settings)
    chunks = make_chunks(3)
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    collection = build_dense_index(
        client, "rag-recursive-embedmap00001", chunks, embeddings, batch_size=10
    )

    for chunk, expected_vector in zip(chunks, embeddings, strict=True):
        result = collection.get(ids=[chunk.chunk_id], include=["embeddings"])
        stored_vector = list(result["embeddings"][0])
        assert stored_vector == pytest.approx(expected_vector)


def test_reopening_persistent_client_can_see_the_records(index_settings: Settings) -> None:
    client = get_chroma_client(index_settings)
    chunks = make_chunks(2)
    embeddings = [[1.0, 2.0], [3.0, 4.0]]
    build_dense_index(client, "rag-recursive-reopen000001", chunks, embeddings, batch_size=10)

    reopened_client = get_chroma_client(index_settings)
    reopened_collection = reopened_client.get_collection(name="rag-recursive-reopen000001")
    assert reopened_collection.count() == 2


def test_verify_dense_collection_passes_for_matching_ids(index_settings: Settings) -> None:
    client = get_chroma_client(index_settings)
    chunks = make_chunks(3)
    embeddings = [[float(i), 0.0] for i in range(3)]
    collection = build_dense_index(
        client, "rag-recursive-verifyok0001", chunks, embeddings, batch_size=10
    )

    verify_dense_collection(collection, [chunk.chunk_id for chunk in chunks])


def test_verify_dense_collection_detects_missing_id(index_settings: Settings) -> None:
    client = get_chroma_client(index_settings)
    chunks = make_chunks(3)
    embeddings = [[float(i), 0.0] for i in range(3)]
    collection = build_dense_index(
        client, "rag-recursive-verifybad0001", chunks, embeddings, batch_size=10
    )

    expected_ids = [chunk.chunk_id for chunk in chunks] + ["missing-chunk-id"]
    with pytest.raises(DenseIndexError):
        verify_dense_collection(collection, expected_ids)


def test_mismatched_chunk_and_embedding_counts_rejected(index_settings: Settings) -> None:
    client = get_chroma_client(index_settings)
    chunks = make_chunks(3)
    with pytest.raises(DenseIndexError):
        build_dense_index(client, "rag-recursive-mismatch0001", chunks, [[1.0, 2.0]], batch_size=10)


def test_chroma_client_can_be_constructed_without_openai_key(index_settings: Settings) -> None:
    # Constructing/using the Chroma client and dense index never touches
    # OpenAI -- precomputed embeddings only.
    assert index_settings.openai_api_key is None
    client = get_chroma_client(index_settings)
    assert isinstance(client, chromadb.ClientAPI)
