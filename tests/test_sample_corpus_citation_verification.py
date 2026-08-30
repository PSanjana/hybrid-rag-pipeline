"""Citation verification over the synthetic Acme Cloud sample corpus (offline).

data/sample -> ingest -> recursive chunking -> indexing -> retrieve_and_generate
(real hybrid + RRF + reranking + generation pipeline, deterministic fake
reranker/generator) -> verify_grounded_answer (deterministic fake judge).
No network/OpenAI calls, and no real cross-encoder or LLM is used
anywhere in this file.

Per the Phase 3 Step 2 spec, these are controlled fake-judge tests --
they prove the verification *pipeline* wires real corpus evidence,
provenance, and citation occurrences together correctly, not that any
judgment logic reflects real LLM semantic quality (that's exhaustively
covered against synthetic data in test_verification.py's semantic
fake-judge tests).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_pipeline.chunking import ChunkingStrategy, chunk_document
from rag_pipeline.config import Settings
from rag_pipeline.generation import (
    CitationVerdict,
    GroundedAnswer,
    extract_citation_occurrences,
    resolve_citation,
    retrieve_and_generate,
    verify_grounded_answer,
)
from rag_pipeline.indexing import index_chunks
from rag_pipeline.ingestion import ingest_document
from rag_pipeline.retrieval import retrieve_hybrid

from .generation.conftest import FakeCitationJudge
from .test_sample_corpus_grounded_generation import _KeywordReranker, _make_keyword_citing_generator
from .test_sample_corpus_retrieval import (
    _SUPPORTED_EXTENSIONS,
    SAMPLE_ROOT,
    ConceptEmbeddingProvider,
)


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


def _make_all_supported_judge(answer: GroundedAnswer) -> FakeCitationJudge:
    """A canned judge that marks every occurrence in `answer` SUPPORTED.

    Used only to prove pipeline wiring (real occurrence extraction, real
    provenance resolution) against a real `GroundedAnswer` -- not a
    claim about judgment quality.
    """
    occurrences = extract_citation_occurrences(answer.answer_text)
    return FakeCitationJudge(
        {
            o.occurrence_id: (o.citation_number, "supported", "matches the cited evidence")
            for o in occurrences
        }
    )


def test_authentication_token_lifetime_citation_verifies_supported(
    pipeline_settings: Settings,
) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    reranker = _KeywordReranker(["token", "mfa"])
    generator = _make_keyword_citing_generator(
        "Access tokens have a limited lifetime.", keywords=["token"]
    )
    answer = retrieve_and_generate(
        "How long do access tokens last?",
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        reranker,
        generator,
        embedding_provider=provider,
    )
    assert answer.cited_numbers

    judge = _make_all_supported_judge(answer)
    report = verify_grounded_answer("How long do access tokens last?", answer, judge)

    assert report.total_occurrences > 0
    assert report.all_supported


def test_production_mfa_citation_verifies_supported(pipeline_settings: Settings) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    reranker = _KeywordReranker(["mfa", "production", "access"])
    generator = _make_keyword_citing_generator("Production access requires MFA.", keywords=["mfa"])
    answer = retrieve_and_generate(
        "Is MFA required for production access?",
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        reranker,
        generator,
        embedding_provider=provider,
    )
    assert answer.cited_numbers

    judge = _make_all_supported_judge(answer)
    report = verify_grounded_answer("Is MFA required for production access?", answer, judge)

    assert report.total_occurrences > 0
    assert report.all_supported


def test_intentionally_altered_token_duration_answer_verifies_contradicted(
    pipeline_settings: Settings,
) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    candidates = retrieve_hybrid(
        "token lifetime",
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        embedding_provider=provider,
    )
    token_hit = next(c for c in candidates if "60 minutes" in c.text.lower())

    from rag_pipeline.generation.models import Evidence

    evidence = (
        Evidence(
            citation_number=1,
            chunk_id=token_hit.chunk_id,
            text=token_hit.text,
            source_file=token_hit.source_file,
            document_id=token_hit.document_id,
            chunk_index=token_hit.chunk_index,
            section_heading=token_hit.section_heading,
            page_number=token_hit.page_number,
            chunking_strategy=token_hit.chunking_strategy,
            reranked_rank=1,
        ),
    )
    # Deliberately wrong: the real evidence says 60 minutes.
    answer = GroundedAnswer(
        answer_text="Access tokens expire after 24 hours [1].",
        evidence=evidence,
        cited_numbers=(1,),
    )

    judge = FakeCitationJudge(
        {1: (1, "contradicted", "Evidence states 60 minutes, claim states 24 hours.")}
    )
    report = verify_grounded_answer("How long do tokens last?", answer, judge)

    assert report.verifications[0].verdict == CitationVerdict.CONTRADICTED
    assert not report.all_supported
    resolved = resolve_citation(answer.evidence, 1)
    assert resolved.source_file in {f.name for f in _sample_files()}


def test_deployment_freeze_fact_verifies_against_deployment_evidence(
    pipeline_settings: Settings,
) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    reranker = _KeywordReranker(["deployment", "freeze"])
    generator = _make_keyword_citing_generator(
        "Deployment freezes reduce risk during sensitive periods.",
        keywords=["freeze", "deployment"],
    )
    answer = retrieve_and_generate(
        "What is a deployment freeze?",
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        reranker,
        generator,
        embedding_provider=provider,
    )
    assert answer.cited_numbers

    judge = _make_all_supported_judge(answer)
    report = verify_grounded_answer("What is a deployment freeze?", answer, judge)

    assert report.total_occurrences > 0
    cited_sources = {resolve_citation(answer.evidence, n).source_file for n in answer.cited_numbers}
    assert any("deploy" in s.lower() or "incident" in s.lower() for s in cited_sources)


def test_err_db_1042_explanation_verifies_against_database_evidence(
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
        "What does ERR_DB_1042 mean?",
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        reranker,
        generator,
        embedding_provider=provider,
    )
    assert answer.cited_numbers

    judge = _make_all_supported_judge(answer)
    report = verify_grounded_answer("What does ERR_DB_1042 mean?", answer, judge)

    assert report.total_occurrences > 0
    cited_sources = {resolve_citation(answer.evidence, n).source_file for n in answer.cited_numbers}
    assert any("database" in s.lower() or "error" in s.lower() for s in cited_sources)


def test_repeated_citation_use_produces_separate_occurrence_judgments(
    pipeline_settings: Settings,
) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    candidates = retrieve_hybrid(
        "token lifetime",
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        embedding_provider=provider,
    )
    token_hit = next(c for c in candidates if "60 minutes" in c.text.lower())

    from rag_pipeline.generation.models import Evidence

    evidence = (
        Evidence(
            citation_number=1,
            chunk_id=token_hit.chunk_id,
            text=token_hit.text,
            source_file=token_hit.source_file,
            document_id=token_hit.document_id,
            chunk_index=token_hit.chunk_index,
            section_heading=token_hit.section_heading,
            page_number=token_hit.page_number,
            chunking_strategy=token_hit.chunking_strategy,
            reranked_rank=1,
        ),
    )
    # [1] is cited twice: once for a claim the evidence supports, once
    # for a claim it does not.
    answer = GroundedAnswer(
        answer_text=(
            "Access tokens expire after 60 minutes [1]. Access tokens also grant "
            "permanent database admin rights [1]."
        ),
        evidence=evidence,
        cited_numbers=(1,),
    )

    judge = FakeCitationJudge(
        {
            1: (1, "supported", "Evidence confirms 60-minute expiry."),
            2: (1, "unsupported", "Evidence says nothing about database admin rights."),
        }
    )
    report = verify_grounded_answer("q", answer, judge)

    assert report.total_occurrences == 2
    verdicts_by_occurrence = {v.occurrence_id: v.verdict for v in report.verifications}
    assert verdicts_by_occurrence[1] == CitationVerdict.SUPPORTED
    assert verdicts_by_occurrence[2] == CitationVerdict.UNSUPPORTED
    # Both occurrences cite the same citation number...
    assert {v.citation_number for v in report.verifications} == {1}
    # ...but received different verdicts, proving occurrences (not
    # citation numbers) are the unit of judgment.
    assert report.verifications[0].verdict != report.verifications[1].verdict


def test_verification_provenance_resolves_to_real_sample_files(
    pipeline_settings: Settings,
) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    reranker = _KeywordReranker(["token", "mfa"])
    generator = _make_keyword_citing_generator(
        "Tokens expire and MFA is required.", keywords=["token", "mfa"]
    )
    answer = retrieve_and_generate(
        "How long do access tokens last, and is MFA required?",
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        reranker,
        generator,
        embedding_provider=provider,
    )
    assert answer.cited_numbers

    judge = _make_all_supported_judge(answer)
    report = verify_grounded_answer(
        "How long do access tokens last, and is MFA required?", answer, judge
    )

    sample_filenames = {f.name for f in _sample_files()}
    for verification in report.verifications:
        resolved = resolve_citation(answer.evidence, verification.citation_number)
        assert resolved.source_file in sample_filenames
        assert verification.chunk_id == resolved.chunk_id
