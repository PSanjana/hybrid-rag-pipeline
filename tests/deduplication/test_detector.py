"""Tests for rag_pipeline.deduplication.detector.deduplicate_chunks."""

from __future__ import annotations

import math

import pytest

from rag_pipeline.deduplication import detector as detector_module
from rag_pipeline.deduplication.detector import deduplicate_chunks
from rag_pipeline.deduplication.exceptions import DeduplicationError
from rag_pipeline.deduplication.models import DuplicateType

from .conftest import make_chunk

# --- core behavior -------------------------------------------------------------


def test_single_chunk_is_always_kept() -> None:
    chunk = make_chunk(text="solo content")
    result = deduplicate_chunks([chunk], [[1.0, 0.0]], threshold=0.95)
    assert result.kept_chunks == (chunk,)
    assert result.duplicates == ()


def test_exact_duplicate_text_is_skipped_without_cosine_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_a = make_chunk(chunk_index=0, text="identical content")
    chunk_b = make_chunk(chunk_index=1, text="identical content")
    assert chunk_a.text == chunk_b.text
    assert chunk_a.chunk_id != chunk_b.chunk_id  # different position -> different id

    def _boom(*_args: object, **_kwargs: object) -> float:
        raise AssertionError("cosine_similarity should not be called for an exact duplicate")

    monkeypatch.setattr(detector_module, "cosine_similarity", _boom)

    result = deduplicate_chunks([chunk_a, chunk_b], [[1.0, 0.0], [0.5, 0.5]], threshold=0.95)

    assert [c.chunk_id for c in result.kept_chunks] == [chunk_a.chunk_id]
    assert len(result.duplicates) == 1
    record = result.duplicates[0]
    assert record.duplicate_type == DuplicateType.EXACT
    assert record.skipped_chunk_id == chunk_b.chunk_id
    assert record.canonical_chunk_id == chunk_a.chunk_id
    assert record.similarity == 1.0


def test_unrelated_vectors_are_both_kept() -> None:
    chunk_a = make_chunk(chunk_index=0, text="alpha")
    chunk_b = make_chunk(chunk_index=1, text="beta, unrelated wording entirely")
    result = deduplicate_chunks([chunk_a, chunk_b], [[1.0, 0.0], [0.0, 1.0]], threshold=0.95)
    assert [c.chunk_id for c in result.kept_chunks] == [chunk_a.chunk_id, chunk_b.chunk_id]
    assert result.duplicates == ()


def test_near_duplicate_above_default_threshold_is_skipped() -> None:
    chunk_a = make_chunk(chunk_index=0, text="alpha content")
    chunk_b = make_chunk(chunk_index=1, text="beta content, nearly identical embedding")
    # Very close vectors: cosine similarity well above 0.95.
    result = deduplicate_chunks(
        [chunk_a, chunk_b], [[1.0, 0.0, 0.0], [0.999, 0.001, 0.0]], threshold=0.95
    )
    assert [c.chunk_id for c in result.kept_chunks] == [chunk_a.chunk_id]
    assert len(result.duplicates) == 1
    assert result.duplicates[0].duplicate_type == DuplicateType.NEAR
    assert result.duplicates[0].similarity > 0.95


def test_similarity_exactly_at_threshold_is_kept_not_flagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_a = make_chunk(chunk_index=0, text="alpha content")
    chunk_b = make_chunk(chunk_index=1, text="beta content, unrelated wording")
    monkeypatch.setattr(detector_module, "cosine_similarity", lambda _a, _b: 0.95)

    result = deduplicate_chunks([chunk_a, chunk_b], [[1.0, 0.0], [0.0, 1.0]], threshold=0.95)

    assert [c.chunk_id for c in result.kept_chunks] == [chunk_a.chunk_id, chunk_b.chunk_id]
    assert result.duplicates == ()


def test_similarity_just_above_threshold_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk_a = make_chunk(chunk_index=0, text="alpha content")
    chunk_b = make_chunk(chunk_index=1, text="beta content, unrelated wording")
    monkeypatch.setattr(detector_module, "cosine_similarity", lambda _a, _b: 0.9500000001)

    result = deduplicate_chunks([chunk_a, chunk_b], [[1.0, 0.0], [0.0, 1.0]], threshold=0.95)

    assert [c.chunk_id for c in result.kept_chunks] == [chunk_a.chunk_id]
    assert len(result.duplicates) == 1
    assert result.duplicates[0].similarity == 0.9500000001


def test_configurable_threshold_changes_behavior() -> None:
    chunk_a = make_chunk(chunk_index=0, text="alpha content")
    chunk_b = make_chunk(chunk_index=1, text="beta content, moderately related")
    embeddings = [[1.0, 0.0], [0.6, 0.8]]  # cosine similarity == 0.6

    lenient = deduplicate_chunks([chunk_a, chunk_b], embeddings, threshold=0.9)
    assert len(lenient.kept_chunks) == 2

    strict = deduplicate_chunks([chunk_a, chunk_b], embeddings, threshold=0.3)
    assert len(strict.kept_chunks) == 1
    assert strict.duplicates[0].duplicate_type == DuplicateType.NEAR


def test_duplicate_maps_to_the_highest_similarity_kept_chunk_and_only_compares_kept() -> None:
    # A at 0 degrees, B at 20 degrees (near-dup of A, threshold 0.9), C at
    # 40 degrees (near-dup of B specifically, but NOT of A) -- if the
    # implementation incorrectly compared C against skipped B, C would be
    # (wrongly) flagged; it must only be compared against the kept set {A}.
    chunk_a = make_chunk(chunk_index=0, text="chunk at 0 degrees")
    chunk_b = make_chunk(chunk_index=1, text="chunk at 20 degrees")
    chunk_c = make_chunk(chunk_index=2, text="chunk at 40 degrees")

    vec_a = [1.0, 0.0]
    vec_b = [math.cos(math.radians(20)), math.sin(math.radians(20))]
    vec_c = [math.cos(math.radians(40)), math.sin(math.radians(40))]

    result = deduplicate_chunks([chunk_a, chunk_b, chunk_c], [vec_a, vec_b, vec_c], threshold=0.9)

    assert [c.chunk_id for c in result.kept_chunks] == [chunk_a.chunk_id, chunk_c.chunk_id]
    assert len(result.duplicates) == 1
    record = result.duplicates[0]
    assert record.skipped_chunk_id == chunk_b.chunk_id
    assert record.canonical_chunk_id == chunk_a.chunk_id


def test_canonical_first_occurrence_is_deterministic_across_a_duplicate_chain() -> None:
    chunk_a = make_chunk(chunk_index=0, text="chunk zero")
    chunk_b = make_chunk(chunk_index=1, text="chunk one")
    chunk_c = make_chunk(chunk_index=2, text="chunk two")
    # All three embeddings identical -> B and C both duplicate A, the
    # first-accepted chunk, never each other.
    embeddings = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]

    result = deduplicate_chunks([chunk_a, chunk_b, chunk_c], embeddings, threshold=0.95)

    assert [c.chunk_id for c in result.kept_chunks] == [chunk_a.chunk_id]
    assert len(result.duplicates) == 2
    assert all(record.canonical_chunk_id == chunk_a.chunk_id for record in result.duplicates)


def test_zero_vector_is_handled_deterministically_not_flagged_as_near_duplicate() -> None:
    chunk_a = make_chunk(chunk_index=0, text="alpha")
    chunk_b = make_chunk(chunk_index=1, text="beta")
    result = deduplicate_chunks([chunk_a, chunk_b], [[0.0, 0.0], [0.0, 0.0]], threshold=0.95)
    assert [c.chunk_id for c in result.kept_chunks] == [chunk_a.chunk_id, chunk_b.chunk_id]
    assert result.duplicates == ()


# --- alignment -------------------------------------------------------------


def test_kept_chunk_count_equals_kept_embedding_count() -> None:
    chunks = [make_chunk(chunk_index=i, text=f"content {i}") for i in range(3)]
    embeddings = [[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    result = deduplicate_chunks(chunks, embeddings, threshold=0.95)
    assert len(result.kept_chunks) == len(result.kept_embeddings) == 3


def test_each_kept_embedding_stays_paired_with_its_original_chunk() -> None:
    chunk_a = make_chunk(chunk_index=0, text="alpha")
    chunk_b = make_chunk(chunk_index=1, text="beta unrelated content")
    result = deduplicate_chunks([chunk_a, chunk_b], [[1.0, 0.0], [0.0, 1.0]], threshold=0.95)
    assert result.kept_chunks == (chunk_a, chunk_b)
    assert result.kept_embeddings == ((1.0, 0.0), (0.0, 1.0))


def test_input_sequences_are_not_mutated() -> None:
    chunks = [make_chunk(chunk_index=0, text="a"), make_chunk(chunk_index=1, text="b")]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    chunks_snapshot = list(chunks)
    embeddings_snapshot = [list(vector) for vector in embeddings]

    deduplicate_chunks(chunks, embeddings, threshold=0.95)

    assert chunks == chunks_snapshot
    assert embeddings == embeddings_snapshot


def test_chunk_embedding_count_mismatch_is_rejected() -> None:
    chunk = make_chunk()
    with pytest.raises(DeduplicationError):
        deduplicate_chunks([chunk], [[1.0, 0.0], [0.0, 1.0]], threshold=0.95)


def test_inconsistent_dimensions_are_rejected() -> None:
    chunk_a = make_chunk(chunk_index=0, text="a")
    chunk_b = make_chunk(chunk_index=1, text="b")
    with pytest.raises(DeduplicationError):
        deduplicate_chunks([chunk_a, chunk_b], [[1.0, 0.0], [1.0, 0.0, 0.0]], threshold=0.95)


def test_empty_vector_is_rejected() -> None:
    chunk = make_chunk()
    with pytest.raises(DeduplicationError):
        deduplicate_chunks([chunk], [[]], threshold=0.95)


def test_non_finite_vector_values_are_rejected() -> None:
    chunk = make_chunk()
    with pytest.raises(DeduplicationError):
        deduplicate_chunks([chunk], [[1.0, float("nan")]], threshold=0.95)


def test_threshold_out_of_range_is_rejected() -> None:
    chunk = make_chunk()
    with pytest.raises(DeduplicationError):
        deduplicate_chunks([chunk], [[1.0, 0.0]], threshold=1.5)
    with pytest.raises(DeduplicationError):
        deduplicate_chunks([chunk], [[1.0, 0.0]], threshold=-0.1)


# --- duplicate records -------------------------------------------------------------


def test_duplicate_record_preserves_source_file_and_document_provenance() -> None:
    chunk_a = make_chunk(
        document_id="a" * 64, chunk_index=0, text="dup text", source_file="first.md"
    )
    chunk_b = make_chunk(
        document_id="b" * 64, chunk_index=0, text="dup text", source_file="second.md"
    )
    result = deduplicate_chunks([chunk_a, chunk_b], [[1.0, 0.0], [0.5, 0.5]], threshold=0.95)

    record = result.duplicates[0]
    assert record.duplicate_type == DuplicateType.EXACT
    assert record.skipped_source_file == "second.md"
    assert record.canonical_source_file == "first.md"
    assert record.skipped_document_id == chunk_b.document_id
    assert record.canonical_document_id == chunk_a.document_id
    assert record.skipped_chunk_index == chunk_b.chunk_index
    assert record.canonical_chunk_index == chunk_a.chunk_index
