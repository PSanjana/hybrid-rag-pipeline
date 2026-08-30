"""Indexes the synthetic Acme Cloud sample corpus end-to-end (offline).

data/sample -> ingest -> recursive chunking -> indexing service ->
Chroma + BM25 snapshot. Uses a deterministic fake embedding provider; makes
no network/OpenAI calls. This validates the corpus + indexing pipeline
together, not retrieval quality (no query API exists yet).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import pytest

from rag_pipeline.chunking import Chunk, ChunkingStrategy, chunk_document
from rag_pipeline.config import Settings
from rag_pipeline.indexing import index_chunks
from rag_pipeline.indexing.dense import get_chroma_client, get_dense_collection_ids
from rag_pipeline.indexing.sparse import load_sparse_snapshot
from rag_pipeline.ingestion import ingest_document

SAMPLE_ROOT = Path(__file__).resolve().parent.parent / "data" / "sample"

REQUIRED_IDENTIFIERS = (
    "err_auth_4017",
    "err_db_1042",
    "err_rate_4290",
    "err_webhook_5003",
    "auth_token_ttl",
    "database_pool_size",
    "database_pool_timeout",
    "max_webhook_retries",
    "deploy_freeze",
)


class FakeEmbeddingProvider:
    """Deterministic, network-free embedding stub.

    Uses the full 32-byte SHA-256 digest, centered to [-1.0, 1.0] rather
    than [0.0, 1.0] -- an all-positive embedding space gives unrelated
    vectors a high baseline cosine similarity (measured mean ~0.76 at 8
    dimensions), risking spurious near-duplicate detection once
    deduplication runs against this fixture's output. Centering + the
    larger dimension keeps max observed similarity among ~300 distinct
    synthetic texts under 0.7, safely below the default 0.95 dedup
    threshold.
    """

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [
            [(byte - 127.5) / 127.5 for byte in hashlib.sha256(text.encode()).digest()]
            for text in texts
        ]


class _ForcedNearDuplicateEmbeddingProvider(FakeEmbeddingProvider):
    """Wraps the base fake provider but forces a chosen set of texts to share one vector."""

    def __init__(self, forced_group: set[str]) -> None:
        self._forced_group = forced_group
        self._forced_vector = super().embed([sorted(forced_group)[0]])[0]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        base = super().embed(texts)
        return [
            list(self._forced_vector) if text in self._forced_group else vector
            for text, vector in zip(texts, base, strict=True)
        ]


@pytest.fixture
def pipeline_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        index_root_dir=tmp_path / "indexes",
    )


_SUPPORTED_EXTENSIONS = {".md", ".txt", ".html", ".pdf"}


def _sample_files() -> list[Path]:
    # Extension-filtered so OS noise (e.g. a Finder-created .DS_Store) can
    # never masquerade as a corpus document.
    return sorted(
        f
        for f in SAMPLE_ROOT.rglob("*")
        if f.is_file() and f.suffix.lower() in _SUPPORTED_EXTENSIONS
    )


def _chunk_sample_corpus(settings: Settings) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in _sample_files():
        document = ingest_document(path, settings=settings)
        chunks.extend(
            chunk_document(document, strategy=ChunkingStrategy.RECURSIVE, settings=settings)
        )
    return chunks


def test_sample_corpus_indexes_successfully_offline(pipeline_settings: Settings) -> None:
    chunks = _chunk_sample_corpus(pipeline_settings)
    assert chunks

    result = index_chunks(chunks, pipeline_settings, embedding_provider=FakeEmbeddingProvider())

    assert result.reused_existing is False
    assert result.manifest.chunk_count == len(chunks)


def test_every_source_chunk_appears_once_in_dense_and_sparse_storage(
    pipeline_settings: Settings,
) -> None:
    chunks = _chunk_sample_corpus(pipeline_settings)
    result = index_chunks(chunks, pipeline_settings, embedding_provider=FakeEmbeddingProvider())

    canonical_ids = [chunk.chunk_id for chunk in chunks]
    assert len(canonical_ids) == len(set(canonical_ids))  # sample corpus produces no duplicates

    client = get_chroma_client(pipeline_settings)
    collection = client.get_collection(name=result.manifest.chroma_collection_name)
    dense_ids = get_dense_collection_ids(collection)
    assert len(dense_ids) == len(canonical_ids)
    assert set(dense_ids) == set(canonical_ids)

    sparse_snapshot = load_sparse_snapshot(pipeline_settings, result.manifest.snapshot_id)
    assert len(sparse_snapshot.chunk_ids) == len(canonical_ids)
    assert set(sparse_snapshot.chunk_ids) == set(canonical_ids)


def test_important_identifiers_survive_the_sparse_snapshot(pipeline_settings: Settings) -> None:
    chunks = _chunk_sample_corpus(pipeline_settings)
    result = index_chunks(chunks, pipeline_settings, embedding_provider=FakeEmbeddingProvider())

    sparse_snapshot = load_sparse_snapshot(pipeline_settings, result.manifest.snapshot_id)
    from rag_pipeline.indexing.tokenizer import tokenize

    all_tokens: set[str] = set()
    for text in sparse_snapshot.texts:
        all_tokens.update(tokenize(text))

    for identifier in REQUIRED_IDENTIFIERS:
        assert identifier in all_tokens, f"missing tokenized identifier: {identifier}"


def test_no_chunk_text_mismatch_between_source_and_dense_storage(
    pipeline_settings: Settings,
) -> None:
    chunks = _chunk_sample_corpus(pipeline_settings)
    result = index_chunks(chunks, pipeline_settings, embedding_provider=FakeEmbeddingProvider())

    client = get_chroma_client(pipeline_settings)
    collection = client.get_collection(name=result.manifest.chroma_collection_name)

    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    stored = collection.get(ids=list(by_id), include=["documents"])
    for chunk_id, document in zip(stored["ids"], stored["documents"], strict=True):
        assert document == by_id[chunk_id].text


# --- deduplication against the real sample corpus -----------------------------


def _find_forced_pair(chunks: list[Chunk]) -> tuple[Chunk, Chunk]:
    candidates = [chunk for chunk in chunks if "ERR_DB_1042" in chunk.text]
    assert len(candidates) >= 2, "sample corpus must contain at least two ERR_DB_1042 chunks"
    target, paraphrase = candidates[0], candidates[1]
    assert target.text != paraphrase.text
    return target, paraphrase


def test_forced_near_duplicate_pair_is_skipped_from_indexing(pipeline_settings: Settings) -> None:
    chunks = _chunk_sample_corpus(pipeline_settings)
    target, paraphrase = _find_forced_pair(chunks)

    provider = _ForcedNearDuplicateEmbeddingProvider({target.text, paraphrase.text})
    result = index_chunks(chunks, pipeline_settings, embedding_provider=provider)

    assert result.manifest.pre_dedup_chunk_count == len(chunks)
    assert result.manifest.duplicate_count >= 1
    assert result.manifest.chunk_count == len(chunks) - result.manifest.duplicate_count

    kept_ids = set(result.manifest.chunk_ids)
    assert (target.chunk_id in kept_ids) != (paraphrase.chunk_id in kept_ids)


def test_required_identifiers_survive_deduplication(pipeline_settings: Settings) -> None:
    chunks = _chunk_sample_corpus(pipeline_settings)
    target, paraphrase = _find_forced_pair(chunks)

    provider = _ForcedNearDuplicateEmbeddingProvider({target.text, paraphrase.text})
    result = index_chunks(chunks, pipeline_settings, embedding_provider=provider)

    sparse_snapshot = load_sparse_snapshot(pipeline_settings, result.manifest.snapshot_id)
    from rag_pipeline.indexing.tokenizer import tokenize

    all_tokens: set[str] = set()
    for text in sparse_snapshot.texts:
        all_tokens.update(tokenize(text))

    for identifier in REQUIRED_IDENTIFIERS:
        assert identifier in all_tokens, f"missing tokenized identifier after dedup: {identifier}"


def test_synchronization_holds_after_deduplication(pipeline_settings: Settings) -> None:
    chunks = _chunk_sample_corpus(pipeline_settings)
    target, paraphrase = _find_forced_pair(chunks)

    provider = _ForcedNearDuplicateEmbeddingProvider({target.text, paraphrase.text})
    result = index_chunks(chunks, pipeline_settings, embedding_provider=provider)

    client = get_chroma_client(pipeline_settings)
    collection = client.get_collection(name=result.manifest.chroma_collection_name)
    dense_ids = set(get_dense_collection_ids(collection))
    sparse_snapshot = load_sparse_snapshot(pipeline_settings, result.manifest.snapshot_id)

    kept_ids = set(result.manifest.chunk_ids)
    assert dense_ids == kept_ids
    assert set(sparse_snapshot.chunk_ids) == kept_ids
