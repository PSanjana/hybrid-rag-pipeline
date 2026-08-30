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
    """Deterministic, network-free embedding stub."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [
            [byte / 255.0 for byte in hashlib.sha256(text.encode()).digest()[:8]] for text in texts
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
