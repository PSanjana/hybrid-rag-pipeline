"""Dense retrieval over the synthetic Acme Cloud sample corpus (offline).

data/sample -> ingest -> recursive chunking -> indexing -> retrieve_dense.

Uses a deterministic, intentionally-engineered fake embedding provider (a
toy bag-of-terms vector -- one dimension per domain-relevant word, plus a
tiny deterministic hash-based perturbation for tie-breaking) rather than
relying on incidental hash-vector similarity. This validates the dense
retrieval *mechanism* (embed -> resolve active snapshot -> query -> rank)
end-to-end against real corpus content, not retrieval quality with a real
embedding model. No network/OpenAI calls.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path

import pytest

from rag_pipeline.chunking import ChunkingStrategy, chunk_document
from rag_pipeline.config import Settings
from rag_pipeline.indexing import index_chunks
from rag_pipeline.indexing.models import IndexManifest
from rag_pipeline.ingestion import ingest_document
from rag_pipeline.retrieval import retrieve_dense

SAMPLE_ROOT = Path(__file__).resolve().parent.parent / "data" / "sample"
_SUPPORTED_EXTENSIONS = {".md", ".txt", ".html", ".pdf"}

# One dimension per term (a toy bag-of-words vector, not a single
# aggregate "topic score"): a chunk's direction only aligns with a query's
# direction when they share several of the *specific* terms, not merely
# any one incidental word -- unlike an aggregate 2-axis score, this isn't
# fooled by cosine similarity's scale-invariance when a chunk contains
# just one incidental hit.
_TERMS = (
    "database",
    "connection",
    "pool",
    "postgres",
    "err_db_1042",
    "authentication",
    "token",
    "login",
    "err_auth_4017",
    "credential",
)
_TERM_PATTERN = re.compile("|".join(rf"\b{re.escape(term)}\b" for term in _TERMS), re.IGNORECASE)
_NOISE_DIMENSIONS = 6


class ConceptEmbeddingProvider:
    """Deterministic, network-free provider producing engineered bag-of-terms vectors.

    Not a real semantic embedding: each text's vector has one dimension
    per domain-relevant term (whole-word count), plus a small deterministic
    hash-based perturbation on further axes for tie-breaking among
    zero-vectors. This is designed intentionally so that a database-themed
    question reliably ranks database-related sample-corpus chunks above
    unrelated ones, and likewise for an auth-themed question -- exercising
    the retrieval pipeline against real corpus content without depending
    on incidental hash-similarity coincidences.
    """

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            term_counts = [0.0] * len(_TERMS)
            for match in _TERM_PATTERN.finditer(text):
                term_counts[_TERMS.index(match.group(0).lower())] += 1.0
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            noise = [(digest[i] - 127.5) / 127.5 * 0.01 for i in range(_NOISE_DIMENSIONS)]
            vectors.append([*term_counts, *noise])
        return vectors


@pytest.fixture
def pipeline_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        index_root_dir=tmp_path / "indexes",
    )


def _sample_files() -> list[Path]:
    return sorted(
        f
        for f in SAMPLE_ROOT.rglob("*")
        if f.is_file() and f.suffix.lower() in _SUPPORTED_EXTENSIONS
    )


def _index_sample_corpus(settings: Settings, provider: ConceptEmbeddingProvider) -> IndexManifest:
    chunks = []
    for path in _sample_files():
        document = ingest_document(path, settings=settings)
        chunks.extend(
            chunk_document(document, strategy=ChunkingStrategy.RECURSIVE, settings=settings)
        )
    result = index_chunks(chunks, settings, embedding_provider=provider)
    return result.manifest


def test_database_question_ranks_database_related_chunks_highly(
    pipeline_settings: Settings,
) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    results = retrieve_dense(
        "What causes the database connection pool to become exhausted?",
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        embedding_provider=provider,
        top_k=5,
    )

    assert results
    top_sources = [r.source_file for r in results[:3]]
    assert any("database" in source.lower() for source in top_sources)


def test_authentication_question_ranks_auth_related_chunks_highly(
    pipeline_settings: Settings,
) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    results = retrieve_dense(
        "How does authentication token expiration work?",
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        embedding_provider=provider,
        top_k=5,
    )

    assert results
    top_sources = [r.source_file for r in results[:3]]
    assert any("auth" in source.lower() for source in top_sources)


def test_result_metadata_points_back_to_real_sample_source_files(
    pipeline_settings: Settings,
) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    results = retrieve_dense(
        "database connection pool",
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        embedding_provider=provider,
        top_k=5,
    )

    sample_filenames = {f.name for f in _sample_files()}
    assert results
    for result in results:
        assert result.source_file in sample_filenames
