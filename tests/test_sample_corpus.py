"""Validates the synthetic Acme Cloud sample corpus under data/sample/.

Narrowly scoped: confirms every sample document ingests successfully, that
fixed/recursive/semantic chunking all respect the configured chunk_size,
and that the corpus contains the identifiers/topics it's meant to (and
omits the ones it's deliberately meant to omit). This is a corpus fixture
check, not an evaluation of retrieval quality.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import pytest

from rag_pipeline.chunking import ChunkingStrategy, chunk_document
from rag_pipeline.config import Settings
from rag_pipeline.ingestion import ingest_document

SAMPLE_ROOT = Path(__file__).resolve().parent.parent / "data" / "sample"

EXPECTED_FILES = {
    "engineering/local-development.md",
    "engineering/deployment-guide.md",
    "engineering/database-operations.md",
    "engineering/api-error-codes.txt",
    "product/authentication-api.md",
    "product/rate-limits.html",
    "product/webhooks.md",
    "operations/incident-response.md",
    "operations/production-runbook.txt",
    "operations/backup-recovery.html",
    "security/access-control-policy.md",
    "people/employee-handbook.pdf",
}

REQUIRED_IDENTIFIERS = (
    "ERR_AUTH_4017",
    "ERR_DB_1042",
    "ERR_RATE_4290",
    "ERR_WEBHOOK_5003",
    "AUTH_TOKEN_TTL",
    "DATABASE_POOL_SIZE",
    "DATABASE_POOL_TIMEOUT",
    "MAX_WEBHOOK_RETRIES",
    "DEPLOY_FREEZE",
)

# These topics must never appear anywhere in the corpus -- they're reserved
# as deliberately-unanswerable questions for later evaluation work.
FORBIDDEN_PHRASES = (
    "vesting",
    "parking",
    "brazil",
)


class FakeEmbeddingProvider:
    """Deterministic, network-free embedding stub for semantic-chunking checks."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vectors.append([byte / 255.0 for byte in digest[:8]])
        return vectors


@pytest.fixture
def corpus_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
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


def test_expected_sample_files_exist() -> None:
    found = {f.relative_to(SAMPLE_ROOT).as_posix() for f in _sample_files()}
    assert found == EXPECTED_FILES


def test_all_four_supported_formats_are_represented() -> None:
    extensions = {f.suffix.lower() for f in _sample_files()}
    assert extensions == {".md", ".txt", ".html", ".pdf"}


def test_every_sample_document_ingests_successfully(corpus_settings: Settings) -> None:
    for path in _sample_files():
        document = ingest_document(path, settings=corpus_settings)
        assert document.segments
        assert all(segment.text.strip() for segment in document.segments)


def test_required_technical_identifiers_are_present() -> None:
    combined_text = "\n".join(path.read_text(errors="ignore") for path in _sample_files())
    for identifier in REQUIRED_IDENTIFIERS:
        assert identifier in combined_text, f"missing identifier: {identifier}"


def test_forbidden_topics_are_absent() -> None:
    combined_text = "\n".join(path.read_text(errors="ignore") for path in _sample_files()).lower()
    for phrase in FORBIDDEN_PHRASES:
        assert phrase not in combined_text, f"forbidden phrase present: {phrase}"


@pytest.mark.parametrize(
    "strategy", [ChunkingStrategy.FIXED, ChunkingStrategy.RECURSIVE, ChunkingStrategy.SEMANTIC]
)
def test_chunking_respects_max_size_across_the_corpus(
    corpus_settings: Settings, strategy: ChunkingStrategy
) -> None:
    embedding_provider = FakeEmbeddingProvider() if strategy == ChunkingStrategy.SEMANTIC else None
    total_chunks = 0
    for path in _sample_files():
        document = ingest_document(path, settings=corpus_settings)
        chunks = chunk_document(
            document,
            strategy=strategy,
            settings=corpus_settings,
            embedding_provider=embedding_provider,
        )
        assert chunks
        assert all(len(chunk.text) <= corpus_settings.chunk_size for chunk in chunks)
        total_chunks += len(chunks)

    # Enough chunks overall to confirm several documents cross chunk
    # boundaries (i.e. the corpus isn't trivially small).
    assert total_chunks > len(EXPECTED_FILES)


def test_some_documents_produce_multiple_chunks(corpus_settings: Settings) -> None:
    multi_chunk_docs = 0
    for path in _sample_files():
        document = ingest_document(path, settings=corpus_settings)
        chunks = chunk_document(document, strategy=ChunkingStrategy.FIXED, settings=corpus_settings)
        if len(chunks) > 1:
            multi_chunk_docs += 1
    assert multi_chunk_docs >= 5
