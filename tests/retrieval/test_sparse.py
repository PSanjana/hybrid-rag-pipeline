"""Tests for rag_pipeline.retrieval.sparse (retrieve_sparse and BM25 ranking/hydration)."""

from __future__ import annotations

import dataclasses
import json

import pytest
from rank_bm25 import BM25Okapi

from rag_pipeline.chunking.models import Chunk
from rag_pipeline.config import ChunkingStrategy, Settings
from rag_pipeline.indexing import index_chunks
from rag_pipeline.indexing.dedup_report import load_dedup_report
from rag_pipeline.indexing.dense import get_chroma_client
from rag_pipeline.indexing.manifest import load_manifest
from rag_pipeline.indexing.sparse import (
    ReconstructedBM25Index,
    load_bm25_index,
    load_sparse_snapshot,
    sparse_corpus_path,
)
from rag_pipeline.indexing.tokenizer import TOKENIZER_VERSION, tokenize
from rag_pipeline.retrieval import sparse as sparse_module
from rag_pipeline.retrieval.exceptions import (
    IndexNotReadyError,
    InvalidQueryError,
    RetrievalError,
    SparseRetrievalError,
    TokenizerVersionMismatchError,
)
from rag_pipeline.retrieval.sparse import (
    _hydrate_results,
    _rank_candidates,
    retrieve_sparse,
)

from .conftest import HashEmbeddingProvider, make_chunk

# --- a standard, real-text corpus used by most tests below ---------------------

_ALPHA_TEXT = "The database connection pool exhausted causing ERR_DB_1042 errors."
_BETA_TEXT = "Authentication tokens expire via AUTH_TOKEN_TTL after thirty minutes."
_GAMMA_TEXT = "Deployment freeze windows are announced well in advance."


def _standard_chunks() -> list[Chunk]:
    return [
        make_chunk(chunk_index=0, text=_ALPHA_TEXT, source_file="alpha.md"),
        make_chunk(chunk_index=1, text=_BETA_TEXT, source_file="beta.md"),
        make_chunk(chunk_index=2, text=_GAMMA_TEXT, source_file="gamma.md"),
    ]


def _index_standard_corpus(settings: Settings) -> None:
    index_chunks(_standard_chunks(), settings, embedding_provider=HashEmbeddingProvider())


# --- query validation --------------------------------------------------------


def test_normal_sparse_query_is_accepted(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    results = retrieve_sparse("ERR_DB_1042", ChunkingStrategy.RECURSIVE, index_settings)
    assert results


def test_empty_query_is_rejected(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    with pytest.raises(InvalidQueryError):
        retrieve_sparse("", ChunkingStrategy.RECURSIVE, index_settings)


def test_whitespace_only_query_is_rejected(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    with pytest.raises(InvalidQueryError):
        retrieve_sparse("   \n\t  ", ChunkingStrategy.RECURSIVE, index_settings)


def test_query_with_no_tokenizer_tokens_is_rejected(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    assert tokenize("??? ... ---") == []
    with pytest.raises(InvalidQueryError):
        retrieve_sparse("??? ... ---", ChunkingStrategy.RECURSIVE, index_settings)


def test_invalid_query_does_not_reconstruct_bm25(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _index_standard_corpus(index_settings)

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("BM25 reconstruction should not happen for an invalid query")

    monkeypatch.setattr(sparse_module, "load_sparse_snapshot", _boom)
    monkeypatch.setattr(sparse_module, "load_bm25_index", _boom)

    with pytest.raises(InvalidQueryError):
        retrieve_sparse("", ChunkingStrategy.RECURSIVE, index_settings)
    with pytest.raises(InvalidQueryError):
        retrieve_sparse("???", ChunkingStrategy.RECURSIVE, index_settings)


# --- active manifest resolution ----------------------------------------------


def test_correct_active_sparse_snapshot_resolves(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    manifest = load_manifest(index_settings, ChunkingStrategy.RECURSIVE)
    assert manifest is not None

    results = retrieve_sparse("database", ChunkingStrategy.RECURSIVE, index_settings)
    returned_ids = {r.chunk_id for r in results}
    assert returned_ids <= set(manifest.chunk_ids)


def test_missing_active_manifest_raises_index_not_ready(index_settings: Settings) -> None:
    with pytest.raises(IndexNotReadyError):
        retrieve_sparse("database", ChunkingStrategy.RECURSIVE, index_settings)


def test_fixed_and_recursive_strategies_resolve_independently(index_settings: Settings) -> None:
    recursive_chunk = make_chunk(
        chunk_index=0, text=_ALPHA_TEXT, source_file="alpha.md", strategy=ChunkingStrategy.RECURSIVE
    )
    fixed_chunk = make_chunk(
        chunk_index=0, text=_BETA_TEXT, source_file="beta.md", strategy=ChunkingStrategy.FIXED
    )
    index_chunks([recursive_chunk], index_settings, embedding_provider=HashEmbeddingProvider())
    index_chunks([fixed_chunk], index_settings, embedding_provider=HashEmbeddingProvider())

    recursive_results = retrieve_sparse("database", ChunkingStrategy.RECURSIVE, index_settings)
    fixed_results = retrieve_sparse("authentication", ChunkingStrategy.FIXED, index_settings)

    assert [r.chunk_id for r in recursive_results] == [recursive_chunk.chunk_id]
    assert [r.chunk_id for r in fixed_results] == [fixed_chunk.chunk_id]


# --- tokenizer -----------------------------------------------------------------


def test_shared_technical_v1_tokenizer_is_used(index_settings: Settings) -> None:
    assert TOKENIZER_VERSION == "technical_v1"
    _index_standard_corpus(index_settings)
    manifest = load_manifest(index_settings, ChunkingStrategy.RECURSIVE)
    assert manifest is not None
    assert manifest.bm25_tokenizer_version == "technical_v1"


def test_tokenizer_version_mismatch_at_manifest_level_is_rejected(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    manifest = load_manifest(index_settings, ChunkingStrategy.RECURSIVE)
    assert manifest is not None

    # Directly exercise the cheap manifest-level check without touching disk.
    bogus_manifest = dataclasses.replace(manifest, bm25_tokenizer_version="some_old_tokenizer_v0")
    with pytest.raises(TokenizerVersionMismatchError):
        sparse_module._check_tokenizer_compatibility(bogus_manifest)


def test_tokenizer_version_mismatch_persisted_on_disk_is_rejected(index_settings: Settings) -> None:
    # Manifest itself claims the current tokenizer (passes the cheap
    # pre-check), but the persisted sparse snapshot file was corrupted to
    # record a different tokenizer_version -- load_sparse_snapshot's own
    # defense-in-depth check must still catch this.
    _index_standard_corpus(index_settings)
    manifest = load_manifest(index_settings, ChunkingStrategy.RECURSIVE)
    assert manifest is not None

    corpus_path = sparse_corpus_path(index_settings, manifest.snapshot_id)
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    data["tokenizer_version"] = "some_old_tokenizer_v0"
    corpus_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(SparseRetrievalError):
        retrieve_sparse("database", ChunkingStrategy.RECURSIVE, index_settings)


def test_err_db_1042_remains_one_token() -> None:
    assert tokenize("ERR_DB_1042") == ["err_db_1042"]


def test_auth_token_ttl_remains_one_token() -> None:
    assert tokenize("AUTH_TOKEN_TTL") == ["auth_token_ttl"]


# --- BM25 reconstruction and scoring --------------------------------------------


def test_bm25_corpus_count_matches_manifest(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    manifest = load_manifest(index_settings, ChunkingStrategy.RECURSIVE)
    assert manifest is not None

    candidates, _texts = sparse_module._score_candidates(index_settings, manifest, ["database"])
    assert len(candidates) == manifest.chunk_count


def test_snapshot_with_substituted_chunk_id_is_rejected(index_settings: Settings) -> None:
    # Same chunk count as the manifest, but one chunk_id has been swapped
    # for an unrelated id -- a corruption that pure count comparison would
    # miss entirely.
    _index_standard_corpus(index_settings)
    manifest = load_manifest(index_settings, ChunkingStrategy.RECURSIVE)
    assert manifest is not None

    corpus_path = sparse_corpus_path(index_settings, manifest.snapshot_id)
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    data["chunk_ids"][0] = "z" * 64
    corpus_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(SparseRetrievalError):
        retrieve_sparse("database", ChunkingStrategy.RECURSIVE, index_settings)


def test_snapshot_with_reordered_chunk_ids_is_rejected(index_settings: Settings) -> None:
    # Exact same chunk_id set, but in a different order than the manifest
    # recorded -- also invisible to a count-only or set-only comparison.
    _index_standard_corpus(index_settings)
    manifest = load_manifest(index_settings, ChunkingStrategy.RECURSIVE)
    assert manifest is not None

    corpus_path = sparse_corpus_path(index_settings, manifest.snapshot_id)
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    assert len(data["chunk_ids"]) >= 2
    data["chunk_ids"] = list(reversed(data["chunk_ids"]))
    corpus_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(SparseRetrievalError):
        retrieve_sparse("database", ChunkingStrategy.RECURSIVE, index_settings)


def test_reconstructed_bm25_chunk_ids_differing_from_manifest_is_rejected(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulates load_bm25_index() itself returning a chunk_id mapping that
    # diverges from the (valid, uncorrupted) snapshot/manifest -- this must
    # be caught independently of the snapshot-vs-manifest check.
    _index_standard_corpus(index_settings)
    manifest = load_manifest(index_settings, ChunkingStrategy.RECURSIVE)
    assert manifest is not None

    real_reconstructed = load_bm25_index(index_settings, manifest.snapshot_id)
    tampered_chunk_ids = tuple(reversed(real_reconstructed.chunk_ids))

    def _tampered_load_bm25_index(*_args: object, **_kwargs: object) -> ReconstructedBM25Index:
        return ReconstructedBM25Index(bm25=real_reconstructed.bm25, chunk_ids=tampered_chunk_ids)

    monkeypatch.setattr(sparse_module, "load_bm25_index", _tampered_load_bm25_index)

    with pytest.raises(SparseRetrievalError):
        retrieve_sparse("database", ChunkingStrategy.RECURSIVE, index_settings)


def test_corpus_position_to_chunk_id_mapping_is_preserved(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    manifest = load_manifest(index_settings, ChunkingStrategy.RECURSIVE)
    assert manifest is not None
    sparse_snapshot = load_sparse_snapshot(index_settings, manifest.snapshot_id)

    candidates, _texts = sparse_module._score_candidates(index_settings, manifest, ["database"])
    for position, chunk_id, _score in candidates:
        assert chunk_id == sparse_snapshot.chunk_ids[position]


def test_exact_identifier_receives_strong_lexical_score(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    results = retrieve_sparse("ERR_DB_1042", ChunkingStrategy.RECURSIVE, index_settings)
    assert results[0].text == _ALPHA_TEXT
    assert results[0].bm25_score > 0.0


def test_ranking_is_descending_by_bm25_score(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    results = retrieve_sparse(
        "database connection pool", ChunkingStrategy.RECURSIVE, index_settings
    )
    scores = [r.bm25_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_rank_starts_at_one_and_is_sequential(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    results = retrieve_sparse("database", ChunkingStrategy.RECURSIVE, index_settings)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_equal_score_ties_preserve_canonical_corpus_position() -> None:
    candidates = [
        (0, "chunk-a", 1.5),
        (1, "chunk-b", 3.0),
        (2, "chunk-c", 3.0),  # ties with chunk-b at position 1
        (3, "chunk-d", 0.5),
    ]
    ranked = _rank_candidates(candidates, top_k=4)
    assert [chunk_id for _pos, chunk_id, _score in ranked] == [
        "chunk-b",
        "chunk-c",
        "chunk-a",
        "chunk-d",
    ]


def test_bm25_reconstructs_a_real_bm25okapi_instance(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    manifest = load_manifest(index_settings, ChunkingStrategy.RECURSIVE)
    assert manifest is not None
    _candidates, _texts = sparse_module._score_candidates(index_settings, manifest, ["database"])
    # Sanity: the reconstruction path really does build a rank_bm25 index,
    # not a hand-rolled scorer.
    reconstructed = load_bm25_index(index_settings, manifest.snapshot_id)
    assert isinstance(reconstructed.bm25, BM25Okapi)


# --- top_k -----------------------------------------------------------------


def test_default_sparse_top_k_is_used(index_settings: Settings) -> None:
    chunks = [
        make_chunk(chunk_index=i, text=f"content number {i} unique", source_file="doc.md")
        for i in range(5)
    ]
    settings = Settings(
        _env_file=None, index_root_dir=index_settings.index_root_dir, sparse_top_k=2
    )
    index_chunks(chunks, settings, embedding_provider=HashEmbeddingProvider())

    results = retrieve_sparse("content", ChunkingStrategy.RECURSIVE, settings)
    assert len(results) == 2


def test_explicit_top_k_override_works(index_settings: Settings) -> None:
    chunks = [
        make_chunk(chunk_index=i, text=f"content number {i} unique", source_file="doc.md")
        for i in range(5)
    ]
    index_chunks(chunks, index_settings, embedding_provider=HashEmbeddingProvider())

    results = retrieve_sparse("content", ChunkingStrategy.RECURSIVE, index_settings, top_k=1)
    assert len(results) == 1


def test_non_positive_top_k_override_is_rejected(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    with pytest.raises(RetrievalError):
        retrieve_sparse("database", ChunkingStrategy.RECURSIVE, index_settings, top_k=0)
    with pytest.raises(RetrievalError):
        retrieve_sparse("database", ChunkingStrategy.RECURSIVE, index_settings, top_k=-1)


def test_top_k_greater_than_corpus_count_returns_available_corpus(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    results = retrieve_sparse("database", ChunkingStrategy.RECURSIVE, index_settings, top_k=1000)
    assert len(results) == 3


# --- results: metadata / provenance --------------------------------------------


def test_metadata_and_provenance_round_trip(index_settings: Settings) -> None:
    markdown_chunk = make_chunk(
        document_id="a" * 64,
        chunk_index=0,
        text="Markdown-sourced database connection pool chunk with a heading.",
        source_file="handbook.md",
        section_heading="Onboarding",
        page_number=None,
    )
    pdf_chunk = make_chunk(
        document_id="b" * 64,
        chunk_index=3,
        text="PDF-sourced database connection pool chunk with a page number.",
        source_file="handbook.pdf",
        section_heading=None,
        page_number=7,
    )
    index_chunks(
        [markdown_chunk, pdf_chunk], index_settings, embedding_provider=HashEmbeddingProvider()
    )

    results = retrieve_sparse(
        "database connection pool", ChunkingStrategy.RECURSIVE, index_settings
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
    assert isinstance(md_result.bm25_score, float)

    pdf_result = by_id[pdf_chunk.chunk_id]
    assert pdf_result.text == pdf_chunk.text
    assert pdf_result.document_id == pdf_chunk.document_id
    assert pdf_result.chunk_index == pdf_chunk.chunk_index
    assert pdf_result.source_file == "handbook.pdf"
    assert pdf_result.section_heading is None
    assert pdf_result.page_number == 7
    assert pdf_result.chunking_strategy == ChunkingStrategy.RECURSIVE


# --- hydration correctness (direct _hydrate_results tests) ---------------------


def _valid_metadata() -> dict:
    return {
        "document_id": "d" * 64,
        "chunk_index": 0,
        "source_file": "doc.md",
        "chunking_strategy": "recursive",
    }


def test_chroma_get_is_used_not_query(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chromadb.api.models.Collection import Collection

    _index_standard_corpus(index_settings)

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Collection.query() should never be called by sparse retrieval")

    monkeypatch.setattr(Collection, "query", _boom)

    results = retrieve_sparse("database", ChunkingStrategy.RECURSIVE, index_settings)
    assert results  # get() succeeded; query() was never touched


def test_chroma_return_order_does_not_change_bm25_rank_order() -> None:
    ranked_candidates = [(0, "c1", 3.0), (1, "c2", 2.0), (2, "c3", 1.0)]
    snapshot_texts_by_id = {"c1": "text1", "c2": "text2", "c3": "text3"}
    # Chroma returns records in the OPPOSITE order from BM25 ranking.
    stored = {
        "ids": ["c3", "c2", "c1"],
        "documents": ["text3", "text2", "text1"],
        "metadatas": [_valid_metadata(), _valid_metadata(), _valid_metadata()],
    }

    results = _hydrate_results(
        stored, ranked_candidates, snapshot_texts_by_id, ChunkingStrategy.RECURSIVE
    )

    assert [r.chunk_id for r in results] == ["c1", "c2", "c3"]
    assert [r.rank for r in results] == [1, 2, 3]


def test_missing_requested_chroma_id_is_detected() -> None:
    ranked_candidates = [(0, "c1", 3.0), (1, "c2", 2.0)]
    snapshot_texts_by_id = {"c1": "text1", "c2": "text2"}
    stored = {
        "ids": ["c1"],  # c2 missing
        "documents": ["text1"],
        "metadatas": [_valid_metadata()],
    }
    with pytest.raises(SparseRetrievalError):
        _hydrate_results(
            stored, ranked_candidates, snapshot_texts_by_id, ChunkingStrategy.RECURSIVE
        )


def test_unexpected_extra_chroma_id_is_detected() -> None:
    ranked_candidates = [(0, "c1", 3.0)]
    snapshot_texts_by_id = {"c1": "text1"}
    stored = {
        "ids": ["c1", "c-unexpected"],
        "documents": ["text1", "text-unexpected"],
        "metadatas": [_valid_metadata(), _valid_metadata()],
    }
    with pytest.raises(SparseRetrievalError):
        _hydrate_results(
            stored, ranked_candidates, snapshot_texts_by_id, ChunkingStrategy.RECURSIVE
        )


def test_sparse_text_vs_chroma_text_mismatch_is_rejected() -> None:
    ranked_candidates = [(0, "c1", 3.0)]
    snapshot_texts_by_id = {"c1": "the real sparse snapshot text"}
    stored = {
        "ids": ["c1"],
        "documents": ["a completely different, corrupted Chroma text"],
        "metadatas": [_valid_metadata()],
    }
    with pytest.raises(SparseRetrievalError):
        _hydrate_results(
            stored, ranked_candidates, snapshot_texts_by_id, ChunkingStrategy.RECURSIVE
        )


def test_malformed_metadata_is_rejected() -> None:
    ranked_candidates = [(0, "c1", 3.0)]
    snapshot_texts_by_id = {"c1": "text1"}
    bad_metadata = _valid_metadata()
    del bad_metadata["document_id"]
    stored = {
        "ids": ["c1"],
        "documents": ["text1"],
        "metadatas": [bad_metadata],
    }
    with pytest.raises(SparseRetrievalError):
        _hydrate_results(
            stored, ranked_candidates, snapshot_texts_by_id, ChunkingStrategy.RECURSIVE
        )


def test_mismatched_chunking_strategy_is_rejected() -> None:
    ranked_candidates = [(0, "c1", 3.0)]
    snapshot_texts_by_id = {"c1": "text1"}
    metadata = _valid_metadata()
    metadata["chunking_strategy"] = "fixed"
    stored = {
        "ids": ["c1"],
        "documents": ["text1"],
        "metadatas": [metadata],
    }
    with pytest.raises(SparseRetrievalError):
        _hydrate_results(
            stored, ranked_candidates, snapshot_texts_by_id, ChunkingStrategy.RECURSIVE
        )


def test_non_string_chroma_get_id_is_rejected() -> None:
    ranked_candidates = [(0, "c1", 3.0)]
    snapshot_texts_by_id = {"c1": "text1"}
    stored = {
        "ids": [123],  # non-string id
        "documents": ["text1"],
        "metadatas": [_valid_metadata()],
    }
    with pytest.raises(SparseRetrievalError):
        _hydrate_results(
            stored, ranked_candidates, snapshot_texts_by_id, ChunkingStrategy.RECURSIVE
        )


def test_empty_string_chroma_get_id_is_rejected() -> None:
    ranked_candidates = [(0, "c1", 3.0)]
    snapshot_texts_by_id = {"c1": "text1"}
    stored = {
        "ids": [""],
        "documents": ["text1"],
        "metadatas": [_valid_metadata()],
    }
    with pytest.raises(SparseRetrievalError):
        _hydrate_results(
            stored, ranked_candidates, snapshot_texts_by_id, ChunkingStrategy.RECURSIVE
        )


def test_duplicate_returned_chroma_id_is_rejected() -> None:
    ranked_candidates = [(0, "c1", 3.0), (1, "c2", 2.0)]
    snapshot_texts_by_id = {"c1": "text1", "c2": "text2"}
    stored = {
        "ids": ["c1", "c1"],  # c1 returned twice; c2 never returned
        "documents": ["text1", "text1"],
        "metadatas": [_valid_metadata(), _valid_metadata()],
    }
    with pytest.raises(SparseRetrievalError, match="duplicate"):
        _hydrate_results(
            stored, ranked_candidates, snapshot_texts_by_id, ChunkingStrategy.RECURSIVE
        )


def test_correct_id_set_but_duplicate_response_cardinality_is_rejected() -> None:
    # The critical regression case: the returned id *set* exactly matches
    # what was requested (a naive `set(ids) == requested_ids` check alone
    # would pass this silently), but the raw response list contains a
    # duplicate record for the single requested id -- the cardinality and
    # duplicate checks must catch this before any set comparison runs.
    ranked_candidates = [(0, "c1", 3.0)]
    snapshot_texts_by_id = {"c1": "text1"}
    stored = {
        "ids": ["c1", "c1"],
        "documents": ["text1", "text1"],
        "metadatas": [_valid_metadata(), _valid_metadata()],
    }
    with pytest.raises(SparseRetrievalError):
        _hydrate_results(
            stored, ranked_candidates, snapshot_texts_by_id, ChunkingStrategy.RECURSIVE
        )


def test_missing_get_result_field_is_rejected() -> None:
    ranked_candidates = [(0, "c1", 3.0)]
    stored = {"ids": ["c1"], "documents": None, "metadatas": [_valid_metadata()]}
    with pytest.raises(SparseRetrievalError):
        _hydrate_results(stored, ranked_candidates, {"c1": "text1"}, ChunkingStrategy.RECURSIVE)


def test_mismatched_get_result_array_lengths_is_rejected() -> None:
    ranked_candidates = [(0, "c1", 3.0)]
    stored = {
        "ids": ["c1"],
        "documents": ["text1", "text2"],  # extra, mismatched length
        "metadatas": [_valid_metadata()],
    }
    with pytest.raises(SparseRetrievalError):
        _hydrate_results(stored, ranked_candidates, {"c1": "text1"}, ChunkingStrategy.RECURSIVE)


# --- scores ------------------------------------------------------------------


def _reindex_with_forced_scores(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch, forced_scores: list[float]
) -> list:
    _index_standard_corpus(index_settings)
    monkeypatch.setattr(BM25Okapi, "get_scores", lambda self, tokens: forced_scores)
    return retrieve_sparse("database", ChunkingStrategy.RECURSIVE, index_settings)


def test_finite_positive_score_is_accepted(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = _reindex_with_forced_scores(index_settings, monkeypatch, [2.5, 0.1, 0.0])
    assert results[0].bm25_score == 2.5


def test_finite_zero_score_is_accepted(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = _reindex_with_forced_scores(index_settings, monkeypatch, [0.0, 0.0, 0.0])
    assert all(r.bm25_score == 0.0 for r in results)


def test_finite_negative_score_is_accepted(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = _reindex_with_forced_scores(index_settings, monkeypatch, [-3.5, -0.1, -9.9])
    scores = {r.bm25_score for r in results}
    assert -3.5 in scores
    assert -0.1 in scores
    assert -9.9 in scores


def test_nan_score_is_rejected(index_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SparseRetrievalError):
        _reindex_with_forced_scores(index_settings, monkeypatch, [float("nan"), 0.0, 0.0])


def test_positive_infinity_score_is_rejected(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SparseRetrievalError):
        _reindex_with_forced_scores(index_settings, monkeypatch, [float("inf"), 0.0, 0.0])


def test_negative_infinity_score_is_rejected(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SparseRetrievalError):
        _reindex_with_forced_scores(index_settings, monkeypatch, [float("-inf"), 0.0, 0.0])


# --- read-only behavior ---------------------------------------------------------


def test_query_does_not_change_dense_record_count(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    manifest = load_manifest(index_settings, ChunkingStrategy.RECURSIVE)
    assert manifest is not None
    client = get_chroma_client(index_settings)
    collection = client.get_collection(name=manifest.chroma_collection_name)
    before_count = collection.count()

    retrieve_sparse("database", ChunkingStrategy.RECURSIVE, index_settings)

    assert collection.count() == before_count


def test_query_does_not_modify_the_manifest(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    before = load_manifest(index_settings, ChunkingStrategy.RECURSIVE)
    assert before is not None

    retrieve_sparse("database", ChunkingStrategy.RECURSIVE, index_settings)

    after = load_manifest(index_settings, ChunkingStrategy.RECURSIVE)
    assert after == before


def test_query_does_not_modify_sparse_snapshot(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    manifest = load_manifest(index_settings, ChunkingStrategy.RECURSIVE)
    assert manifest is not None
    before = load_sparse_snapshot(index_settings, manifest.snapshot_id)

    retrieve_sparse("database", ChunkingStrategy.RECURSIVE, index_settings)

    after = load_sparse_snapshot(index_settings, manifest.snapshot_id)
    assert after == before


def test_query_does_not_modify_dedup_report(index_settings: Settings) -> None:
    _index_standard_corpus(index_settings)
    manifest = load_manifest(index_settings, ChunkingStrategy.RECURSIVE)
    assert manifest is not None
    before = load_dedup_report(index_settings, manifest.snapshot_id)

    retrieve_sparse("database", ChunkingStrategy.RECURSIVE, index_settings)

    after = load_dedup_report(index_settings, manifest.snapshot_id)
    assert after == before
