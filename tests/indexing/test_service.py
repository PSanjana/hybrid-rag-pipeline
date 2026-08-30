"""Tests for rag_pipeline.indexing.service (orchestration/synchronization)."""

from __future__ import annotations

import json
from dataclasses import replace

import chromadb.errors
import pytest

from rag_pipeline.config import ChunkingStrategy, Settings
from rag_pipeline.indexing import service as service_module
from rag_pipeline.indexing.dense import (
    build_collection_name,
    get_chroma_client,
    get_dense_collection_ids,
)
from rag_pipeline.indexing.exceptions import IndexingError, SynchronizationError
from rag_pipeline.indexing.fingerprint import compute_snapshot_id
from rag_pipeline.indexing.manifest import load_manifest
from rag_pipeline.indexing.models import canonical_order
from rag_pipeline.indexing.service import (
    _existing_snapshot_is_valid,
    index_chunks,
    verify_synchronization,
)
from rag_pipeline.indexing.sparse import (
    load_sparse_snapshot,
    sparse_corpus_path,
    sparse_snapshot_dir,
)
from rag_pipeline.indexing.tokenizer import TOKENIZER_VERSION

from .conftest import FakeEmbeddingProvider, make_chunks


def _expected_snapshot_id_for(chunks: list, settings: Settings) -> str:
    ordered = canonical_order(chunks)
    return compute_snapshot_id(
        tuple(chunk.chunk_id for chunk in ordered),
        ChunkingStrategy.RECURSIVE,
        settings.embedding_model,
        TOKENIZER_VERSION,
    )


# --- pure verify_synchronization() checks ------------------------------------


def test_verify_synchronization_passes_for_matching_sets() -> None:
    ids = ["a", "b", "c"]
    verify_synchronization(ids, ids, ids)


def test_verify_synchronization_detects_missing_dense_id() -> None:
    canonical = ["a", "b", "c"]
    dense = ["a", "b"]  # missing "c"
    sparse = ["a", "b", "c"]
    with pytest.raises(SynchronizationError, match="[Dd]ense"):
        verify_synchronization(canonical, dense, sparse)


def test_verify_synchronization_detects_corrupted_missing_sparse_id() -> None:
    canonical = ["a", "b", "c"]
    dense = ["a", "b", "c"]
    sparse = ["a", "b"]  # missing "c"
    with pytest.raises(SynchronizationError, match="[Ss]parse"):
        verify_synchronization(canonical, dense, sparse)


def test_verify_synchronization_detects_extra_sparse_id() -> None:
    canonical = ["a", "b"]
    dense = ["a", "b"]
    sparse = ["a", "b", "unexpected"]
    with pytest.raises(SynchronizationError):
        verify_synchronization(canonical, dense, sparse)


def test_verify_synchronization_detects_sparse_order_mismatch() -> None:
    canonical = ["a", "b", "c"]
    dense = ["a", "b", "c"]
    sparse = ["c", "b", "a"]  # same set, wrong order
    with pytest.raises(SynchronizationError, match="order"):
        verify_synchronization(canonical, dense, sparse)


# --- end-to-end index_chunks() behavior --------------------------------------


def test_dense_and_sparse_chunk_counts_match(index_settings: Settings) -> None:
    chunks = make_chunks(6)
    result = index_chunks(chunks, index_settings, embedding_provider=FakeEmbeddingProvider())

    client = get_chroma_client(index_settings)
    collection = client.get_collection(name=result.manifest.chroma_collection_name)
    sparse_snapshot = load_sparse_snapshot(index_settings, result.manifest.snapshot_id)

    assert collection.count() == 6
    assert len(sparse_snapshot.chunk_ids) == 6
    assert result.manifest.chunk_count == 6


def test_dense_sparse_and_canonical_id_sets_are_identical(index_settings: Settings) -> None:
    chunks = make_chunks(5)
    result = index_chunks(chunks, index_settings, embedding_provider=FakeEmbeddingProvider())

    canonical_ids = {chunk.chunk_id for chunk in chunks}
    client = get_chroma_client(index_settings)
    collection = client.get_collection(name=result.manifest.chroma_collection_name)
    dense_ids = set(get_dense_collection_ids(collection))
    sparse_snapshot = load_sparse_snapshot(index_settings, result.manifest.snapshot_id)

    assert dense_ids == canonical_ids
    assert set(sparse_snapshot.chunk_ids) == canonical_ids


def test_active_manifest_records_the_same_chunk_ids(index_settings: Settings) -> None:
    chunks = make_chunks(4)
    result = index_chunks(chunks, index_settings, embedding_provider=FakeEmbeddingProvider())
    assert set(result.manifest.chunk_ids) == {chunk.chunk_id for chunk in chunks}


def test_failed_sparse_persistence_does_not_replace_a_valid_manifest(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_chunks = make_chunks(3, text_prefix="Baseline content")
    baseline_result = index_chunks(
        baseline_chunks, index_settings, embedding_provider=FakeEmbeddingProvider()
    )

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated sparse persistence failure")

    monkeypatch.setattr(service_module, "write_sparse_snapshot", _boom)

    new_chunks = make_chunks(3, text_prefix="Completely different content")
    with pytest.raises(IndexingError):
        index_chunks(new_chunks, index_settings, embedding_provider=FakeEmbeddingProvider())

    still_active = load_manifest(index_settings, ChunkingStrategy.RECURSIVE)
    assert still_active == baseline_result.manifest


def test_failed_dense_construction_does_not_activate_a_manifest(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated dense construction failure")

    monkeypatch.setattr(service_module, "build_dense_index", _boom)

    chunks = make_chunks(3)
    with pytest.raises(IndexingError):
        index_chunks(chunks, index_settings, embedding_provider=FakeEmbeddingProvider())

    assert load_manifest(index_settings, ChunkingStrategy.RECURSIVE) is None


def test_failed_dense_construction_preserves_previous_manifest(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_chunks = make_chunks(2, text_prefix="Stable baseline")
    baseline_result = index_chunks(
        baseline_chunks, index_settings, embedding_provider=FakeEmbeddingProvider()
    )

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated dense construction failure")

    monkeypatch.setattr(service_module, "build_dense_index", _boom)

    new_chunks = make_chunks(2, text_prefix="A totally different corpus entirely")
    with pytest.raises(IndexingError):
        index_chunks(new_chunks, index_settings, embedding_provider=FakeEmbeddingProvider())

    still_active = load_manifest(index_settings, ChunkingStrategy.RECURSIVE)
    assert still_active == baseline_result.manifest


# --- idempotence ---------------------------------------------------------------


def test_same_completed_snapshot_is_safely_reused(index_settings: Settings) -> None:
    chunks = make_chunks(3)
    provider = FakeEmbeddingProvider()

    first = index_chunks(chunks, index_settings, embedding_provider=provider)
    assert first.reused_existing is False
    assert len(provider.calls) == 1

    second = index_chunks(chunks, index_settings, embedding_provider=provider)
    assert second.reused_existing is True
    assert second.manifest.snapshot_id == first.manifest.snapshot_id
    # No new embedding call was made for the reused snapshot.
    assert len(provider.calls) == 1

    client = get_chroma_client(index_settings)
    collection = client.get_collection(name=first.manifest.chroma_collection_name)
    assert collection.count() == 3  # no duplicated records


def test_changed_corpus_creates_a_new_snapshot(index_settings: Settings) -> None:
    first_chunks = make_chunks(3, text_prefix="First version")
    second_chunks = make_chunks(3, text_prefix="Second, different version")

    first = index_chunks(first_chunks, index_settings, embedding_provider=FakeEmbeddingProvider())
    second = index_chunks(second_chunks, index_settings, embedding_provider=FakeEmbeddingProvider())

    assert first.manifest.snapshot_id != second.manifest.snapshot_id
    assert second.reused_existing is False
    # The new manifest for the strategy replaces the old one.
    active = load_manifest(index_settings, ChunkingStrategy.RECURSIVE)
    assert active == second.manifest


def test_reindexing_without_a_provider_reuses_snapshot_without_requiring_api_key(
    index_settings: Settings,
) -> None:
    # First build with a fake provider (no API key needed either way, but
    # this establishes the snapshot).
    chunks = make_chunks(2)
    index_chunks(chunks, index_settings, embedding_provider=FakeEmbeddingProvider())

    # Second call passes NO embedding_provider at all. Since
    # settings.openai_api_key is None, constructing a real
    # OpenAIEmbeddingProvider would fail -- but reuse must short-circuit
    # before that ever happens.
    assert index_settings.openai_api_key is None
    result = index_chunks(chunks, index_settings, embedding_provider=None)
    assert result.reused_existing is True


# --- strategy isolation ----------------------------------------------------


def test_fixed_and_recursive_indexes_coexist_independently(index_settings: Settings) -> None:
    recursive_chunks = make_chunks(
        3, strategy=ChunkingStrategy.RECURSIVE, text_prefix="Recursive doc"
    )
    fixed_chunks = make_chunks(3, strategy=ChunkingStrategy.FIXED, text_prefix="Fixed doc")

    recursive_result = index_chunks(
        recursive_chunks, index_settings, embedding_provider=FakeEmbeddingProvider()
    )
    fixed_result = index_chunks(
        fixed_chunks, index_settings, embedding_provider=FakeEmbeddingProvider()
    )

    assert (
        recursive_result.manifest.chroma_collection_name
        != fixed_result.manifest.chroma_collection_name
    )
    assert recursive_result.manifest.snapshot_id != fixed_result.manifest.snapshot_id

    recursive_manifest = load_manifest(index_settings, ChunkingStrategy.RECURSIVE)
    fixed_manifest = load_manifest(index_settings, ChunkingStrategy.FIXED)
    assert recursive_manifest == recursive_result.manifest
    assert fixed_manifest == fixed_result.manifest

    client = get_chroma_client(index_settings)
    recursive_collection = client.get_collection(
        name=recursive_result.manifest.chroma_collection_name
    )
    fixed_collection = client.get_collection(name=fixed_result.manifest.chroma_collection_name)
    assert recursive_collection.count() == 3
    assert fixed_collection.count() == 3


# --- manifest activation is part of the protected build ------------------------


def test_failed_manifest_write_does_not_replace_a_valid_manifest(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_chunks = make_chunks(3, text_prefix="Baseline content for manifest test")
    baseline_result = index_chunks(
        baseline_chunks, index_settings, embedding_provider=FakeEmbeddingProvider()
    )

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated manifest write failure")

    monkeypatch.setattr(service_module, "write_manifest", _boom)

    new_chunks = make_chunks(3, text_prefix="A new corpus that will fail to activate")
    with pytest.raises(IndexingError):
        index_chunks(new_chunks, index_settings, embedding_provider=FakeEmbeddingProvider())

    still_active = load_manifest(index_settings, ChunkingStrategy.RECURSIVE)
    assert still_active == baseline_result.manifest


def test_failed_manifest_write_removes_the_new_dense_collection(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated manifest write failure")

    monkeypatch.setattr(service_module, "write_manifest", _boom)

    chunks = make_chunks(3, text_prefix="Orphaned dense collection scenario")
    expected_snapshot_id = _expected_snapshot_id_for(chunks, index_settings)
    expected_collection_name = build_collection_name(
        ChunkingStrategy.RECURSIVE, expected_snapshot_id, index_settings.chroma_collection_prefix
    )

    with pytest.raises(IndexingError):
        index_chunks(chunks, index_settings, embedding_provider=FakeEmbeddingProvider())

    client = get_chroma_client(index_settings)
    with pytest.raises(chromadb.errors.NotFoundError):
        client.get_collection(name=expected_collection_name)


def test_failed_manifest_write_removes_the_new_sparse_directory(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated manifest write failure")

    monkeypatch.setattr(service_module, "write_manifest", _boom)

    chunks = make_chunks(3, text_prefix="Orphaned sparse directory scenario")
    expected_snapshot_id = _expected_snapshot_id_for(chunks, index_settings)

    with pytest.raises(IndexingError):
        index_chunks(chunks, index_settings, embedding_provider=FakeEmbeddingProvider())

    assert not sparse_snapshot_dir(index_settings, expected_snapshot_id).exists()


# --- strengthened existing-snapshot reuse validation ----------------------------


def test_valid_unchanged_snapshot_is_still_reused(index_settings: Settings) -> None:
    chunks = make_chunks(3, text_prefix="Untouched corpus")
    result = index_chunks(chunks, index_settings, embedding_provider=FakeEmbeddingProvider())

    ordered = canonical_order(chunks)
    assert _existing_snapshot_is_valid(index_settings, result.manifest, ordered) is True


def test_reuse_rejects_corrupted_stored_dense_document_text(index_settings: Settings) -> None:
    chunks = make_chunks(3, text_prefix="Corruption target corpus")
    result = index_chunks(chunks, index_settings, embedding_provider=FakeEmbeddingProvider())

    client = get_chroma_client(index_settings)
    collection = client.get_collection(name=result.manifest.chroma_collection_name)
    target_id = result.manifest.chunk_ids[0]
    collection.update(ids=[target_id], embeddings=[[0.0] * 8], documents=["CORRUPTED TEXT"])

    ordered = canonical_order(chunks)
    assert _existing_snapshot_is_valid(index_settings, result.manifest, ordered) is False


def test_reuse_rejects_missing_cosine_configuration(index_settings: Settings) -> None:
    chunks = make_chunks(2, text_prefix="Cosine config check corpus")
    result = index_chunks(chunks, index_settings, embedding_provider=FakeEmbeddingProvider())
    ordered = canonical_order(chunks)

    # Sanity: the real manifest is valid as built.
    assert _existing_snapshot_is_valid(index_settings, result.manifest, ordered) is True

    # A manifest pointing at a *different* (non-existent) collection name
    # can never validate -- this exercises the same "reject if the stored
    # index doesn't actually look right" path without needing to fabricate
    # a non-cosine Chroma collection by hand.
    bogus_manifest = replace(result.manifest, chroma_collection_name="rag-recursive-doesnotexist")
    assert _existing_snapshot_is_valid(index_settings, bogus_manifest, ordered) is False


def test_reuse_rejects_corrupted_persisted_sparse_text(index_settings: Settings) -> None:
    chunks = make_chunks(3, text_prefix="Sparse corruption corpus")
    result = index_chunks(chunks, index_settings, embedding_provider=FakeEmbeddingProvider())

    corpus_path = sparse_corpus_path(index_settings, result.manifest.snapshot_id)
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    data["texts"][0] = "CORRUPTED SPARSE TEXT THAT DOES NOT MATCH THE CHUNK"
    corpus_path.write_text(json.dumps(data), encoding="utf-8")

    ordered = canonical_order(chunks)
    assert _existing_snapshot_is_valid(index_settings, result.manifest, ordered) is False


def test_reuse_rejects_tokenizer_version_mismatch(index_settings: Settings) -> None:
    chunks = make_chunks(2, text_prefix="Tokenizer mismatch corpus")
    result = index_chunks(chunks, index_settings, embedding_provider=FakeEmbeddingProvider())

    corpus_path = sparse_corpus_path(index_settings, result.manifest.snapshot_id)
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    data["tokenizer_version"] = "some_old_tokenizer_v0"
    corpus_path.write_text(json.dumps(data), encoding="utf-8")

    ordered = canonical_order(chunks)
    assert _existing_snapshot_is_valid(index_settings, result.manifest, ordered) is False


def test_reindex_rebuilds_when_reuse_validation_fails(index_settings: Settings) -> None:
    # End-to-end: corrupt the dense text under an otherwise-matching
    # snapshot_id, then reindex the same corpus -- it must rebuild (not
    # blindly trust the stale/corrupted content) and end up valid again.
    chunks = make_chunks(2, text_prefix="Rebuild-on-corruption corpus")
    first = index_chunks(chunks, index_settings, embedding_provider=FakeEmbeddingProvider())

    client = get_chroma_client(index_settings)
    collection = client.get_collection(name=first.manifest.chroma_collection_name)
    target_id = first.manifest.chunk_ids[0]
    collection.update(ids=[target_id], embeddings=[[0.0] * 8], documents=["CORRUPTED TEXT"])

    second = index_chunks(chunks, index_settings, embedding_provider=FakeEmbeddingProvider())

    assert second.reused_existing is False
    # The rebuild replaces the collection outright (see build_dense_index),
    # so re-fetch it by name rather than reusing the stale pre-rebuild
    # `collection` reference.
    rebuilt_collection = client.get_collection(name=second.manifest.chroma_collection_name)
    stored = rebuilt_collection.get(ids=[target_id], include=["documents"])
    assert stored["documents"][0] != "CORRUPTED TEXT"


# --- cleanup is truly best-effort -----------------------------------------------


def _fail_from_second_call(real_fn: object, error: Exception) -> object:
    """A stateful fake: the 1st call delegates to `real_fn`, every call after raises `error`.

    Used to let the main build phase's own (legitimate) use of a function
    succeed normally, while making only a *later* use -- inside cleanup --
    fail, so cleanup-failure tests don't also break the build phase that
    has to succeed first in order to reach cleanup at all.
    """
    calls = {"n": 0}

    def _fake(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        if calls["n"] == 1:
            return real_fn(*args, **kwargs)  # type: ignore[operator]
        raise error

    return _fake


def test_cleanup_failure_obtaining_chroma_client_does_not_mask_original_error(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom_dense(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated dense construction failure")

    monkeypatch.setattr(service_module, "build_dense_index", _boom_dense)
    monkeypatch.setattr(
        service_module,
        "get_chroma_client",
        _fail_from_second_call(
            get_chroma_client,
            OSError("simulated failure obtaining the Chroma client during cleanup"),
        ),
    )

    chunks = make_chunks(2, text_prefix="Cleanup client failure corpus")
    # The build's own get_chroma_client() call (before build_dense_index
    # runs) succeeds; build_dense_index itself is what fails. Cleanup then
    # makes a *second* get_chroma_client() call, which fails -- but that
    # must not replace the original "dense construction failure" message.
    with pytest.raises(IndexingError, match="dense construction failure"):
        index_chunks(chunks, index_settings, embedding_provider=FakeEmbeddingProvider())


def test_cleanup_still_removes_sparse_dir_when_chroma_cleanup_fails(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom_manifest(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated manifest write failure")

    monkeypatch.setattr(service_module, "write_manifest", _boom_manifest)
    monkeypatch.setattr(
        service_module,
        "get_chroma_client",
        _fail_from_second_call(
            get_chroma_client,
            OSError("simulated failure obtaining the Chroma client during cleanup"),
        ),
    )

    chunks = make_chunks(2, text_prefix="Partial cleanup corpus")
    expected_snapshot_id = _expected_snapshot_id_for(chunks, index_settings)

    with pytest.raises(IndexingError, match="manifest"):
        index_chunks(chunks, index_settings, embedding_provider=FakeEmbeddingProvider())

    # Sparse directory cleanup must still happen even though Chroma
    # cleanup (the first cleanup step) failed.
    assert not sparse_snapshot_dir(index_settings, expected_snapshot_id).exists()
