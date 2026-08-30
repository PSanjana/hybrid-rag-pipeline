"""Tests for rag_pipeline.retrieval.dense (retrieve_dense and response parsing)."""

from __future__ import annotations

import dataclasses

import chromadb.errors
import pytest
from chromadb.api.models.Collection import Collection

from rag_pipeline.chunking.models import Chunk
from rag_pipeline.config import ChunkingStrategy, Settings
from rag_pipeline.embeddings.exceptions import EmbeddingProviderError
from rag_pipeline.embeddings.similarity import cosine_similarity
from rag_pipeline.indexing import index_chunks
from rag_pipeline.indexing.dedup_report import load_dedup_report
from rag_pipeline.indexing.dense import get_chroma_client
from rag_pipeline.indexing.manifest import load_manifest
from rag_pipeline.indexing.sparse import load_sparse_snapshot
from rag_pipeline.retrieval.dense import _parse_query_response, retrieve_dense
from rag_pipeline.retrieval.exceptions import (
    DenseRetrievalError,
    EmbeddingModelMismatchError,
    IndexNotReadyError,
    InvalidQueryError,
    RetrievalError,
)

from .conftest import DictEmbeddingProvider, make_chunk

# --- a standard, hand-engineered 2D corpus used by most tests below -------------

_ALPHA_TEXT = "Alpha content about widgets."
_BETA_TEXT = "Beta content about gadgets."
_GAMMA_TEXT = "Gamma content about gizmos."
_QUERY_NEAR_ALPHA = "a question about widgets"

_STANDARD_VECTORS = {
    _ALPHA_TEXT: [1.0, 0.0],
    _BETA_TEXT: [0.0, 1.0],
    _GAMMA_TEXT: [-1.0, 0.0],
    _QUERY_NEAR_ALPHA: [0.9, 0.1],
}


def _standard_chunks() -> list[Chunk]:
    return [
        make_chunk(chunk_index=0, text=_ALPHA_TEXT, source_file="alpha.md"),
        make_chunk(chunk_index=1, text=_BETA_TEXT, source_file="beta.md"),
        make_chunk(chunk_index=2, text=_GAMMA_TEXT, source_file="gamma.md"),
    ]


def _index_standard_corpus(settings: Settings) -> DictEmbeddingProvider:
    provider = DictEmbeddingProvider(_STANDARD_VECTORS)
    index_chunks(_standard_chunks(), settings, embedding_provider=provider)
    return provider


# --- query validation --------------------------------------------------------


def test_normal_question_is_accepted(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    provider = DictEmbeddingProvider(_STANDARD_VECTORS)
    results = retrieve_dense(
        _QUERY_NEAR_ALPHA, ChunkingStrategy.RECURSIVE, index_settings, embedding_provider=provider
    )
    assert results


def test_empty_question_is_rejected(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    with pytest.raises(InvalidQueryError):
        retrieve_dense("", ChunkingStrategy.RECURSIVE, index_settings, embedding_provider=None)


def test_whitespace_only_question_is_rejected(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    with pytest.raises(InvalidQueryError):
        retrieve_dense(
            "   \n\t  ", ChunkingStrategy.RECURSIVE, index_settings, embedding_provider=None
        )


def test_invalid_question_does_not_call_embedding_provider(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)

    class _BoomProvider:
        def embed(self, texts: object) -> object:
            raise AssertionError("embed() should not be called for an invalid query")

    with pytest.raises(InvalidQueryError):
        retrieve_dense(
            "", ChunkingStrategy.RECURSIVE, index_settings, embedding_provider=_BoomProvider()
        )


# --- active manifest resolution ----------------------------------------------


def test_missing_active_manifest_raises_clear_error(index_settings: Settings) -> None:
    provider = DictEmbeddingProvider(_STANDARD_VECTORS)
    with pytest.raises(IndexNotReadyError):
        retrieve_dense(
            _QUERY_NEAR_ALPHA,
            ChunkingStrategy.RECURSIVE,
            index_settings,
            embedding_provider=provider,
        )


def test_fixed_and_recursive_strategies_retrieve_from_their_own_snapshots(
    index_settings: Settings,
) -> None:
    recursive_text = "Recursive strategy content about widgets."
    fixed_text = "Fixed strategy content about widgets."
    query_text = "widgets question"
    vectors = {
        recursive_text: [1.0, 0.0],
        fixed_text: [1.0, 0.0],
        query_text: [1.0, 0.0],
    }

    recursive_chunk = make_chunk(
        chunk_index=0, text=recursive_text, strategy=ChunkingStrategy.RECURSIVE
    )
    fixed_chunk = make_chunk(chunk_index=0, text=fixed_text, strategy=ChunkingStrategy.FIXED)

    index_chunks(
        [recursive_chunk], index_settings, embedding_provider=DictEmbeddingProvider(vectors)
    )
    index_chunks([fixed_chunk], index_settings, embedding_provider=DictEmbeddingProvider(vectors))

    recursive_results = retrieve_dense(
        query_text,
        ChunkingStrategy.RECURSIVE,
        index_settings,
        embedding_provider=DictEmbeddingProvider(vectors),
    )
    fixed_results = retrieve_dense(
        query_text,
        ChunkingStrategy.FIXED,
        index_settings,
        embedding_provider=DictEmbeddingProvider(vectors),
    )

    assert [r.chunk_id for r in recursive_results] == [recursive_chunk.chunk_id]
    assert [r.chunk_id for r in fixed_results] == [fixed_chunk.chunk_id]


# --- embeddings ---------------------------------------------------------------


def test_query_is_embedded_exactly_once(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    provider = DictEmbeddingProvider(_STANDARD_VECTORS)
    retrieve_dense(
        _QUERY_NEAR_ALPHA, ChunkingStrategy.RECURSIVE, index_settings, embedding_provider=provider
    )
    assert provider.calls == [[_QUERY_NEAR_ALPHA]]


def test_embedding_provider_failure_is_wrapped_as_dense_retrieval_error(
    index_settings: Settings,
) -> None:
    _index_standard_corpus(index_settings)

    class _FailingProvider:
        def embed(self, texts: object) -> list[list[float]]:
            raise EmbeddingProviderError("simulated embedding provider failure")

    with pytest.raises(
        DenseRetrievalError, match="simulated embedding provider failure"
    ) as exc_info:
        retrieve_dense(
            _QUERY_NEAR_ALPHA,
            ChunkingStrategy.RECURSIVE,
            index_settings,
            embedding_provider=_FailingProvider(),
        )
    assert isinstance(exc_info.value.__cause__, EmbeddingProviderError)


def test_exactly_one_vector_is_required(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)

    class _TwoVectorProvider:
        def embed(self, texts: object) -> list[list[float]]:
            return [[1.0, 0.0], [0.0, 1.0]]

    with pytest.raises(DenseRetrievalError):
        retrieve_dense(
            _QUERY_NEAR_ALPHA,
            ChunkingStrategy.RECURSIVE,
            index_settings,
            embedding_provider=_TwoVectorProvider(),
        )


def test_empty_query_vector_is_rejected(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)

    class _EmptyVectorProvider:
        def embed(self, texts: object) -> list[list[float]]:
            return [[]]

    with pytest.raises(DenseRetrievalError):
        retrieve_dense(
            _QUERY_NEAR_ALPHA,
            ChunkingStrategy.RECURSIVE,
            index_settings,
            embedding_provider=_EmptyVectorProvider(),
        )


def test_non_finite_query_vector_is_rejected(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)

    class _NonFiniteVectorProvider:
        def embed(self, texts: object) -> list[list[float]]:
            return [[1.0, float("nan")]]

    with pytest.raises(DenseRetrievalError):
        retrieve_dense(
            _QUERY_NEAR_ALPHA,
            ChunkingStrategy.RECURSIVE,
            index_settings,
            embedding_provider=_NonFiniteVectorProvider(),
        )


def test_query_vector_dimension_mismatch_is_rejected(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)  # built with 2-dim vectors

    class _WrongDimensionProvider:
        def embed(self, texts: object) -> list[list[float]]:
            return [[1.0, 0.0, 0.0]]

    with pytest.raises(DenseRetrievalError):
        retrieve_dense(
            _QUERY_NEAR_ALPHA,
            ChunkingStrategy.RECURSIVE,
            index_settings,
            embedding_provider=_WrongDimensionProvider(),
        )


def test_embedding_model_mismatch_is_rejected_before_embedding(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)  # built with settings.embedding_model default

    mismatched_settings = Settings(
        _env_file=None,
        index_root_dir=index_settings.index_root_dir,
        embedding_model="a-completely-different-model",
    )

    class _BoomProvider:
        def embed(self, texts: object) -> object:
            raise AssertionError("embed() should not be called on a model mismatch")

    with pytest.raises(EmbeddingModelMismatchError):
        retrieve_dense(
            _QUERY_NEAR_ALPHA,
            ChunkingStrategy.RECURSIVE,
            mismatched_settings,
            embedding_provider=_BoomProvider(),
        )


def test_no_api_key_required_when_fake_provider_supplied(index_settings: Settings) -> None:
    assert index_settings.openai_api_key is None
    _index_standard_corpus(index_settings)
    provider = DictEmbeddingProvider(_STANDARD_VECTORS)
    results = retrieve_dense(
        _QUERY_NEAR_ALPHA, ChunkingStrategy.RECURSIVE, index_settings, embedding_provider=provider
    )
    assert results


# --- dense results: ordering, ranking, top_k ----------------------------------


def test_top_result_corresponds_to_nearest_fake_embedding(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    provider = DictEmbeddingProvider(_STANDARD_VECTORS)
    results = retrieve_dense(
        _QUERY_NEAR_ALPHA, ChunkingStrategy.RECURSIVE, index_settings, embedding_provider=provider
    )
    assert results[0].text == _ALPHA_TEXT


def test_returned_order_matches_chroma_ranking_by_increasing_distance(
    index_settings: Settings,
) -> None:
    _index_standard_corpus(index_settings)
    provider = DictEmbeddingProvider(_STANDARD_VECTORS)
    results = retrieve_dense(
        _QUERY_NEAR_ALPHA, ChunkingStrategy.RECURSIVE, index_settings, embedding_provider=provider
    )
    assert [r.text for r in results] == [_ALPHA_TEXT, _BETA_TEXT, _GAMMA_TEXT]
    assert [r.distance for r in results] == sorted(r.distance for r in results)


def test_rank_starts_at_one_and_is_sequential(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    provider = DictEmbeddingProvider(_STANDARD_VECTORS)
    results = retrieve_dense(
        _QUERY_NEAR_ALPHA, ChunkingStrategy.RECURSIVE, index_settings, embedding_provider=provider
    )
    assert [r.rank for r in results] == [1, 2, 3]


def test_top_k_limits_results(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    provider = DictEmbeddingProvider(_STANDARD_VECTORS)
    results = retrieve_dense(
        _QUERY_NEAR_ALPHA,
        ChunkingStrategy.RECURSIVE,
        index_settings,
        embedding_provider=provider,
        top_k=2,
    )
    assert len(results) == 2


def test_top_k_greater_than_corpus_size_returns_available_corpus(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    provider = DictEmbeddingProvider(_STANDARD_VECTORS)
    results = retrieve_dense(
        _QUERY_NEAR_ALPHA,
        ChunkingStrategy.RECURSIVE,
        index_settings,
        embedding_provider=provider,
        top_k=1000,
    )
    assert len(results) == 3


def test_default_top_k_comes_from_settings(index_settings: Settings) -> None:
    chunks = [
        make_chunk(chunk_index=i, text=f"content {i}", source_file="doc.md") for i in range(5)
    ]
    vectors = {chunk.text: [float(i), 0.0] for i, chunk in enumerate(chunks)}
    vectors["query"] = [0.0, 0.0]
    settings = Settings(_env_file=None, index_root_dir=index_settings.index_root_dir, dense_top_k=2)
    index_chunks(chunks, settings, embedding_provider=DictEmbeddingProvider(vectors))

    results = retrieve_dense(
        "query",
        ChunkingStrategy.RECURSIVE,
        settings,
        embedding_provider=DictEmbeddingProvider(vectors),
    )
    assert len(results) == 2


def test_explicit_top_k_override_works(index_settings: Settings) -> None:
    chunks = [
        make_chunk(chunk_index=i, text=f"content {i}", source_file="doc.md") for i in range(5)
    ]
    vectors = {chunk.text: [float(i), 0.0] for i, chunk in enumerate(chunks)}
    vectors["query"] = [0.0, 0.0]
    index_chunks(chunks, index_settings, embedding_provider=DictEmbeddingProvider(vectors))

    results = retrieve_dense(
        "query",
        ChunkingStrategy.RECURSIVE,
        index_settings,
        embedding_provider=DictEmbeddingProvider(vectors),
        top_k=1,
    )
    assert len(results) == 1


def test_invalid_top_k_is_rejected(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    provider = DictEmbeddingProvider(_STANDARD_VECTORS)

    with pytest.raises(RetrievalError):
        retrieve_dense(
            _QUERY_NEAR_ALPHA,
            ChunkingStrategy.RECURSIVE,
            index_settings,
            embedding_provider=provider,
            top_k=0,
        )
    with pytest.raises(RetrievalError):
        retrieve_dense(
            _QUERY_NEAR_ALPHA,
            ChunkingStrategy.RECURSIVE,
            index_settings,
            embedding_provider=provider,
            top_k=-1,
        )
    assert provider.calls == []  # rejected before ever embedding


# --- metadata / provenance ----------------------------------------------------


def test_metadata_and_provenance_round_trip(index_settings: Settings) -> None:
    markdown_chunk = make_chunk(
        document_id="a" * 64,
        chunk_index=0,
        text="Markdown-sourced chunk with a heading.",
        source_file="handbook.md",
        section_heading="Onboarding",
        page_number=None,
    )
    pdf_chunk = make_chunk(
        document_id="b" * 64,
        chunk_index=3,
        text="PDF-sourced chunk with a page number.",
        source_file="handbook.pdf",
        section_heading=None,
        page_number=7,
    )
    vectors = {
        markdown_chunk.text: [1.0, 0.0],
        pdf_chunk.text: [0.0, 1.0],
        "query": [1.0, 0.0],
    }
    index_chunks(
        [markdown_chunk, pdf_chunk],
        index_settings,
        embedding_provider=DictEmbeddingProvider(vectors),
    )

    results = retrieve_dense(
        "query",
        ChunkingStrategy.RECURSIVE,
        index_settings,
        embedding_provider=DictEmbeddingProvider(vectors),
    )
    by_id = {r.chunk_id: r for r in results}

    md_result = by_id[markdown_chunk.chunk_id]
    assert md_result.text == markdown_chunk.text
    assert md_result.document_id == markdown_chunk.document_id
    assert md_result.chunk_index == markdown_chunk.chunk_index
    assert md_result.source_file == "handbook.md"
    assert md_result.section_heading == "Onboarding"
    assert md_result.page_number is None
    assert md_result.chunking_strategy == ChunkingStrategy.RECURSIVE

    pdf_result = by_id[pdf_chunk.chunk_id]
    assert pdf_result.text == pdf_chunk.text
    assert pdf_result.document_id == pdf_chunk.document_id
    assert pdf_result.chunk_index == pdf_chunk.chunk_index
    assert pdf_result.source_file == "handbook.pdf"
    assert pdf_result.section_heading is None
    assert pdf_result.page_number == 7
    assert pdf_result.chunking_strategy == ChunkingStrategy.RECURSIVE


# --- scoring -------------------------------------------------------------------


def test_raw_distance_matches_cosine_distance_of_the_engineered_vectors(
    index_settings: Settings,
) -> None:
    _index_standard_corpus(index_settings)
    provider = DictEmbeddingProvider(_STANDARD_VECTORS)
    results = retrieve_dense(
        _QUERY_NEAR_ALPHA, ChunkingStrategy.RECURSIVE, index_settings, embedding_provider=provider
    )

    query_vector = _STANDARD_VECTORS[_QUERY_NEAR_ALPHA]
    for result in results:
        chunk_vector = _STANDARD_VECTORS[result.text]
        expected_distance = 1.0 - cosine_similarity(query_vector, chunk_vector)
        assert result.distance == pytest.approx(expected_distance, abs=1e-4)


def test_similarity_equals_one_minus_distance(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    provider = DictEmbeddingProvider(_STANDARD_VECTORS)
    results = retrieve_dense(
        _QUERY_NEAR_ALPHA, ChunkingStrategy.RECURSIVE, index_settings, embedding_provider=provider
    )
    for result in results:
        assert result.similarity == pytest.approx(1.0 - result.distance)


def test_rank_order_corresponds_to_increasing_distance(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    provider = DictEmbeddingProvider(_STANDARD_VECTORS)
    results = retrieve_dense(
        _QUERY_NEAR_ALPHA, ChunkingStrategy.RECURSIVE, index_settings, embedding_provider=provider
    )
    distances = [r.distance for r in results]
    assert distances == sorted(distances)
    ranks = [r.rank for r in results]
    assert ranks == sorted(ranks)


# --- corruption / malformed-response handling (direct _parse_query_response tests) --


def _valid_metadata(chunk_id: str = "c1") -> dict:
    return {
        "document_id": "d" * 64,
        "chunk_index": 0,
        "source_file": "doc.md",
        "chunking_strategy": "recursive",
    }


def test_missing_document_field_is_rejected() -> None:
    raw = {
        "ids": [["c1"]],
        "documents": [[None]],
        "metadatas": [[_valid_metadata()]],
        "distances": [[0.1]],
    }
    with pytest.raises(DenseRetrievalError):
        _parse_query_response(raw, ChunkingStrategy.RECURSIVE)


def test_missing_metadata_field_is_rejected() -> None:
    raw = {
        "ids": [["c1"]],
        "documents": [["text"]],
        "metadatas": [[None]],
        "distances": [[0.1]],
    }
    with pytest.raises(DenseRetrievalError):
        _parse_query_response(raw, ChunkingStrategy.RECURSIVE)


def test_metadata_missing_required_key_is_rejected() -> None:
    bad_metadata = _valid_metadata()
    del bad_metadata["document_id"]
    raw = {
        "ids": [["c1"]],
        "documents": [["text"]],
        "metadatas": [[bad_metadata]],
        "distances": [[0.1]],
    }
    with pytest.raises(DenseRetrievalError):
        _parse_query_response(raw, ChunkingStrategy.RECURSIVE)


def test_metadata_with_wrong_type_is_rejected() -> None:
    bad_metadata = _valid_metadata()
    bad_metadata["chunk_index"] = "not-an-int"
    raw = {
        "ids": [["c1"]],
        "documents": [["text"]],
        "metadatas": [[bad_metadata]],
        "distances": [[0.1]],
    }
    with pytest.raises(DenseRetrievalError):
        _parse_query_response(raw, ChunkingStrategy.RECURSIVE)


def test_mismatched_result_array_lengths_are_rejected() -> None:
    raw = {
        "ids": [["c1", "c2"]],
        "documents": [["text1"]],  # only one document for two ids
        "metadatas": [[_valid_metadata(), _valid_metadata()]],
        "distances": [[0.1, 0.2]],
    }
    with pytest.raises(DenseRetrievalError):
        _parse_query_response(raw, ChunkingStrategy.RECURSIVE)


def test_multiple_query_batches_are_rejected() -> None:
    raw = {
        "ids": [["c1"], ["c2"]],
        "documents": [["text1"], ["text2"]],
        "metadatas": [[_valid_metadata()], [_valid_metadata()]],
        "distances": [[0.1], [0.2]],
    }
    with pytest.raises(DenseRetrievalError):
        _parse_query_response(raw, ChunkingStrategy.RECURSIVE)


def test_zero_document_batches_is_rejected() -> None:
    raw = {
        "ids": [["c1"]],
        "documents": [],
        "metadatas": [[_valid_metadata()]],
        "distances": [[0.1]],
    }
    with pytest.raises(DenseRetrievalError):
        _parse_query_response(raw, ChunkingStrategy.RECURSIVE)


def test_two_document_batches_is_rejected() -> None:
    raw = {
        "ids": [["c1"]],
        "documents": [["text1"], ["text2"]],
        "metadatas": [[_valid_metadata()]],
        "distances": [[0.1]],
    }
    with pytest.raises(DenseRetrievalError):
        _parse_query_response(raw, ChunkingStrategy.RECURSIVE)


def test_zero_metadata_batches_is_rejected() -> None:
    raw = {
        "ids": [["c1"]],
        "documents": [["text1"]],
        "metadatas": [],
        "distances": [[0.1]],
    }
    with pytest.raises(DenseRetrievalError):
        _parse_query_response(raw, ChunkingStrategy.RECURSIVE)


def test_multiple_metadata_batches_is_rejected() -> None:
    raw = {
        "ids": [["c1"]],
        "documents": [["text1"]],
        "metadatas": [[_valid_metadata()], [_valid_metadata()]],
        "distances": [[0.1]],
    }
    with pytest.raises(DenseRetrievalError):
        _parse_query_response(raw, ChunkingStrategy.RECURSIVE)


def test_zero_distance_batches_is_rejected() -> None:
    raw = {
        "ids": [["c1"]],
        "documents": [["text1"]],
        "metadatas": [[_valid_metadata()]],
        "distances": [],
    }
    with pytest.raises(DenseRetrievalError):
        _parse_query_response(raw, ChunkingStrategy.RECURSIVE)


def test_multiple_distance_batches_is_rejected() -> None:
    raw = {
        "ids": [["c1"]],
        "documents": [["text1"]],
        "metadatas": [[_valid_metadata()]],
        "distances": [[0.1], [0.2]],
    }
    with pytest.raises(DenseRetrievalError):
        _parse_query_response(raw, ChunkingStrategy.RECURSIVE)


def test_empty_string_id_is_rejected() -> None:
    raw = {
        "ids": [[""]],
        "documents": [["text"]],
        "metadatas": [[_valid_metadata()]],
        "distances": [[0.1]],
    }
    with pytest.raises(DenseRetrievalError):
        _parse_query_response(raw, ChunkingStrategy.RECURSIVE)


def test_non_string_id_is_rejected() -> None:
    raw = {
        "ids": [[123]],
        "documents": [["text"]],
        "metadatas": [[_valid_metadata()]],
        "distances": [[0.1]],
    }
    with pytest.raises(DenseRetrievalError):
        _parse_query_response(raw, ChunkingStrategy.RECURSIVE)


@pytest.mark.parametrize("bad_distance", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_distance_is_rejected(bad_distance: float) -> None:
    raw = {
        "ids": [["c1"]],
        "documents": [["text"]],
        "metadatas": [[_valid_metadata()]],
        "distances": [[bad_distance]],
    }
    with pytest.raises(DenseRetrievalError):
        _parse_query_response(raw, ChunkingStrategy.RECURSIVE)


def test_section_heading_absent_is_none() -> None:
    raw = {
        "ids": [["c1"]],
        "documents": [["text"]],
        "metadatas": [[_valid_metadata()]],  # no section_heading key at all
        "distances": [[0.1]],
    }
    results = _parse_query_response(raw, ChunkingStrategy.RECURSIVE)
    assert results[0].section_heading is None


def test_section_heading_present_with_wrong_type_is_rejected() -> None:
    bad_metadata = _valid_metadata()
    bad_metadata["section_heading"] = 42
    raw = {
        "ids": [["c1"]],
        "documents": [["text"]],
        "metadatas": [[bad_metadata]],
        "distances": [[0.1]],
    }
    with pytest.raises(DenseRetrievalError):
        _parse_query_response(raw, ChunkingStrategy.RECURSIVE)


def test_page_number_absent_is_none() -> None:
    raw = {
        "ids": [["c1"]],
        "documents": [["text"]],
        "metadatas": [[_valid_metadata()]],  # no page_number key at all
        "distances": [[0.1]],
    }
    results = _parse_query_response(raw, ChunkingStrategy.RECURSIVE)
    assert results[0].page_number is None


def test_page_number_present_with_wrong_type_is_rejected() -> None:
    bad_metadata = _valid_metadata()
    bad_metadata["page_number"] = "not-an-int"
    raw = {
        "ids": [["c1"]],
        "documents": [["text"]],
        "metadatas": [[bad_metadata]],
        "distances": [[0.1]],
    }
    with pytest.raises(DenseRetrievalError):
        _parse_query_response(raw, ChunkingStrategy.RECURSIVE)


def test_page_number_present_as_bool_is_rejected() -> None:
    bad_metadata = _valid_metadata()
    bad_metadata["page_number"] = True
    raw = {
        "ids": [["c1"]],
        "documents": [["text"]],
        "metadatas": [[bad_metadata]],
        "distances": [[0.1]],
    }
    with pytest.raises(DenseRetrievalError):
        _parse_query_response(raw, ChunkingStrategy.RECURSIVE)


def test_section_heading_and_page_number_present_and_valid_are_returned() -> None:
    good_metadata = _valid_metadata()
    good_metadata["section_heading"] = "Intro"
    good_metadata["page_number"] = 3
    raw = {
        "ids": [["c1"]],
        "documents": [["text"]],
        "metadatas": [[good_metadata]],
        "distances": [[0.1]],
    }
    results = _parse_query_response(raw, ChunkingStrategy.RECURSIVE)
    assert results[0].section_heading == "Intro"
    assert results[0].page_number == 3


def test_missing_result_field_is_rejected() -> None:
    raw = {
        "ids": [["c1"]],
        "documents": None,
        "metadatas": [[_valid_metadata()]],
        "distances": [[0.1]],
    }
    with pytest.raises(DenseRetrievalError):
        _parse_query_response(raw, ChunkingStrategy.RECURSIVE)


def test_missing_metadatas_or_distances_field_is_rejected() -> None:
    raw = {
        "ids": [["c1"]],
        "documents": [["text"]],
        "metadatas": None,
        "distances": [[0.1]],
    }
    with pytest.raises(DenseRetrievalError):
        _parse_query_response(raw, ChunkingStrategy.RECURSIVE)


def test_non_numeric_distance_is_rejected() -> None:
    raw = {
        "ids": [["c1"]],
        "documents": [["text"]],
        "metadatas": [[_valid_metadata()]],
        "distances": [["not-a-number"]],
    }
    with pytest.raises(DenseRetrievalError):
        _parse_query_response(raw, ChunkingStrategy.RECURSIVE)


def test_invalid_chunking_strategy_value_is_rejected() -> None:
    metadata = _valid_metadata()
    metadata["chunking_strategy"] = "not-a-real-strategy"
    raw = {
        "ids": [["c1"]],
        "documents": [["text"]],
        "metadatas": [[metadata]],
        "distances": [[0.1]],
    }
    with pytest.raises(DenseRetrievalError):
        _parse_query_response(raw, ChunkingStrategy.RECURSIVE)


def test_mismatched_chunking_strategy_is_rejected() -> None:
    metadata = _valid_metadata()
    metadata["chunking_strategy"] = "fixed"
    raw = {
        "ids": [["c1"]],
        "documents": [["text"]],
        "metadatas": [[metadata]],
        "distances": [[0.1]],
    }
    with pytest.raises(DenseRetrievalError):
        _parse_query_response(raw, ChunkingStrategy.RECURSIVE)


def test_chroma_query_failure_produces_clear_retrieval_error(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _index_standard_corpus(index_settings)

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise chromadb.errors.ChromaError("simulated Chroma query failure")

    monkeypatch.setattr(Collection, "query", _boom)

    provider = DictEmbeddingProvider(_STANDARD_VECTORS)
    with pytest.raises(DenseRetrievalError):
        retrieve_dense(
            _QUERY_NEAR_ALPHA,
            ChunkingStrategy.RECURSIVE,
            index_settings,
            embedding_provider=provider,
        )


def test_missing_collection_produces_clear_retrieval_error(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    manifest = load_manifest(index_settings, ChunkingStrategy.RECURSIVE)
    assert manifest is not None

    client = get_chroma_client(index_settings)
    client.delete_collection(name=manifest.chroma_collection_name)

    provider = DictEmbeddingProvider(_STANDARD_VECTORS)
    with pytest.raises(DenseRetrievalError):
        retrieve_dense(
            _QUERY_NEAR_ALPHA,
            ChunkingStrategy.RECURSIVE,
            index_settings,
            embedding_provider=provider,
        )


# --- read-only behavior ---------------------------------------------------------


def test_query_does_not_change_dense_record_count(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    manifest = load_manifest(index_settings, ChunkingStrategy.RECURSIVE)
    assert manifest is not None
    client = get_chroma_client(index_settings)
    collection = client.get_collection(name=manifest.chroma_collection_name)
    before_count = collection.count()

    provider = DictEmbeddingProvider(_STANDARD_VECTORS)
    retrieve_dense(
        _QUERY_NEAR_ALPHA, ChunkingStrategy.RECURSIVE, index_settings, embedding_provider=provider
    )

    assert collection.count() == before_count


def test_query_does_not_modify_the_manifest(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    before = load_manifest(index_settings, ChunkingStrategy.RECURSIVE)
    assert before is not None

    provider = DictEmbeddingProvider(_STANDARD_VECTORS)
    retrieve_dense(
        _QUERY_NEAR_ALPHA, ChunkingStrategy.RECURSIVE, index_settings, embedding_provider=provider
    )

    after = load_manifest(index_settings, ChunkingStrategy.RECURSIVE)
    assert after == before


def test_query_does_not_modify_sparse_snapshot_or_dedup_report(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    manifest = load_manifest(index_settings, ChunkingStrategy.RECURSIVE)
    assert manifest is not None
    sparse_before = load_sparse_snapshot(index_settings, manifest.snapshot_id)
    dedup_before = load_dedup_report(index_settings, manifest.snapshot_id)

    provider = DictEmbeddingProvider(_STANDARD_VECTORS)
    retrieve_dense(
        _QUERY_NEAR_ALPHA, ChunkingStrategy.RECURSIVE, index_settings, embedding_provider=provider
    )

    sparse_after = load_sparse_snapshot(index_settings, manifest.snapshot_id)
    dedup_after = load_dedup_report(index_settings, manifest.snapshot_id)
    assert sparse_after == sparse_before
    assert dedup_after == dedup_before


def test_dense_retrieval_result_is_frozen(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    provider = DictEmbeddingProvider(_STANDARD_VECTORS)
    results = retrieve_dense(
        _QUERY_NEAR_ALPHA, ChunkingStrategy.RECURSIVE, index_settings, embedding_provider=provider
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        results[0].rank = 99  # type: ignore[misc]
