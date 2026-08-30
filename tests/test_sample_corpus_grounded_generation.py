"""Grounded generation over the synthetic Acme Cloud sample corpus (offline).

data/sample -> ingest -> recursive chunking -> indexing -> retrieve_and_generate
(real hybrid + RRF + reranking pipeline, deterministic fake reranker and
fake generation provider). No network/OpenAI calls, and no real
cross-encoder or LLM is used.

The fake generator (`_make_keyword_citing_generator`) simulates "the
model picked out the relevant evidence and cited it" by scanning the
rendered evidence block for keyword matches -- a deterministic stand-in
for real LLM judgment, not a claim that this is how the production
provider behaves.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import pytest

from rag_pipeline.chunking import ChunkingStrategy, chunk_document
from rag_pipeline.config import Settings
from rag_pipeline.generation import resolve_citation, retrieve_and_generate
from rag_pipeline.indexing import index_chunks
from rag_pipeline.ingestion import ingest_document

from .generation.conftest import FakeGenerator
from .test_sample_corpus_retrieval import (
    _SUPPORTED_EXTENSIONS,
    SAMPLE_ROOT,
    ConceptEmbeddingProvider,
)

_EVIDENCE_BLOCK_PATTERN = re.compile(r"\[(\d+)\]\n(.*?)(?=\n\[\d+\]\n|\Z)", re.DOTALL)


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


def _index_sample_corpus(settings: Settings, provider: ConceptEmbeddingProvider) -> None:
    chunks = []
    for path in _sample_files():
        document = ingest_document(path, settings=settings)
        chunks.extend(
            chunk_document(document, strategy=ChunkingStrategy.RECURSIVE, settings=settings)
        )
    index_chunks(chunks, settings, embedding_provider=provider)


class _KeywordReranker:
    """Deterministic `Reranker`: score = keyword hit count in each candidate's own text."""

    def __init__(self, keywords: Sequence[str]) -> None:
        self._keywords = [k.lower() for k in keywords]

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        return [
            float(sum(document.lower().count(keyword) for keyword in self._keywords))
            for document in documents
        ]


def _make_keyword_citing_generator(claim: str, keywords: Sequence[str]) -> FakeGenerator:
    """A fake generator that cites every evidence block containing one of `keywords`.

    Deterministically simulates "the model grounded its claim in the
    relevant evidence" by scanning the rendered evidence block (built by
    `generation.context.format_evidence_block`) for keyword matches --
    if none match, it returns the fixed insufficient-evidence sentence
    instead of inventing an answer.
    """

    def response_fn(system_prompt: str, user_prompt: str) -> str:
        cited = [
            int(match.group(1))
            for match in _EVIDENCE_BLOCK_PATTERN.finditer(user_prompt)
            if any(keyword.lower() in match.group(2).lower() for keyword in keywords)
        ]
        if not cited:
            return (
                "The supplied documents do not provide enough information to answer this question."
            )
        citation_suffix = "".join(f"[{n}]" for n in cited)
        return f"{claim} {citation_suffix}"

    return FakeGenerator(response_fn=response_fn)


def test_authentication_question_cites_token_lifetime_and_mfa_evidence(
    pipeline_settings: Settings,
) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    reranker = _KeywordReranker(["token", "mfa", "authentication"])
    generator = _make_keyword_citing_generator(
        "Access tokens have a limited lifetime and production access requires MFA.",
        keywords=["token", "mfa"],
    )

    answer = retrieve_and_generate(
        "How long do access tokens last, and is MFA required for production access?",
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        reranker,
        generator,
        embedding_provider=provider,
    )

    assert answer.cited_numbers
    sample_filenames = {f.name for f in _sample_files()}
    for number in answer.cited_numbers:
        resolved = resolve_citation(answer.evidence, number)
        assert resolved.source_file in sample_filenames


def test_deployment_freeze_question_cites_deployment_or_incident_evidence(
    pipeline_settings: Settings,
) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    reranker = _KeywordReranker(["deployment", "freeze", "incident"])
    generator = _make_keyword_citing_generator(
        "Deployment freezes are used to reduce risk during sensitive periods.",
        keywords=["freeze", "deployment"],
    )

    answer = retrieve_and_generate(
        "What is a deployment freeze and when is one used?",
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        reranker,
        generator,
        embedding_provider=provider,
    )

    assert answer.cited_numbers
    cited_sources = {resolve_citation(answer.evidence, n).source_file for n in answer.cited_numbers}
    assert any("deploy" in s.lower() or "incident" in s.lower() for s in cited_sources)


def test_database_troubleshooting_answer_cites_database_or_error_code_evidence(
    pipeline_settings: Settings,
) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    reranker = _KeywordReranker(["database", "err_db_1042", "pool"])
    generator = _make_keyword_citing_generator(
        "ERR_DB_1042 indicates connection pool exhaustion.",
        keywords=["err_db_1042", "pool"],
    )

    answer = retrieve_and_generate(
        "What does ERR_DB_1042 mean and how do I troubleshoot it?",
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        reranker,
        generator,
        embedding_provider=provider,
    )

    assert answer.cited_numbers
    cited_sources = {resolve_citation(answer.evidence, n).source_file for n in answer.cited_numbers}
    assert any("database" in s.lower() or "error" in s.lower() for s in cited_sources)


def test_citation_numbers_resolve_back_to_real_sample_corpus_files(
    pipeline_settings: Settings,
) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    reranker = _KeywordReranker(["token", "mfa"])
    generator = _make_keyword_citing_generator(
        "Access tokens expire and MFA is required.", keywords=["token", "mfa"]
    )

    answer = retrieve_and_generate(
        "How long do access tokens last, and is MFA required for production access?",
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        reranker,
        generator,
        embedding_provider=provider,
    )

    sample_filenames = {f.name for f in _sample_files()}
    assert answer.cited_numbers
    for number in answer.cited_numbers:
        resolved = resolve_citation(answer.evidence, number)
        assert resolved.source_file in sample_filenames
        assert resolved.chunk_id


def test_absent_topic_question_produces_explicit_insufficient_evidence_response(
    pipeline_settings: Settings,
) -> None:
    # "office parking" is not covered anywhere in the synthetic Acme Cloud
    # corpus -- the keyword-citing fake generator therefore finds nothing
    # to cite and falls back to the fixed insufficient-evidence sentence,
    # exactly as prompt.SYSTEM_PROMPT rule 6 instructs a real model to.
    # This only proves the generation instruction/response *form* exists;
    # it is NOT the formal confidence/abstention system (that's a later
    # phase) -- a real LLM could still answer differently.
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    reranker = _KeywordReranker(["office", "parking"])
    generator = _make_keyword_citing_generator(
        "Employees may park in the garage.", keywords=["office parking", "reserved spot"]
    )

    answer = retrieve_and_generate(
        "How many office parking spots are reserved for visitors?",
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        reranker,
        generator,
        embedding_provider=provider,
    )

    assert answer.cited_numbers == ()
    assert "do not provide enough information" in answer.answer_text.lower()
