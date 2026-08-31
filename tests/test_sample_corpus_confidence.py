"""Deterministic confidence scoring over the synthetic Acme Cloud sample corpus (offline).

data/sample -> ingest -> recursive chunking -> indexing -> retrieve_reranked
(real hybrid + RRF + reranking pipeline, deterministic fake reranker) ->
generate_grounded_answer (deterministic keyword-citing fake generator) ->
verify_grounded_answer (deterministic fake judge) -> score_confidence.

No network/OpenAI calls, and no real cross-encoder or LLM is used
anywhere in this file. These tests exercise the confidence *pipeline*
against real corpus evidence/provenance/retrieval diagnostics; the
judge verdicts are controlled fakes, so the numeric scores here are a
property of the fixed verdict/agreement inputs, NOT an empirically
calibrated probability that any answer is correct.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from rag_pipeline.chunking import ChunkingStrategy, chunk_document
from rag_pipeline.config import Settings
from rag_pipeline.generation import (
    GroundedAnswer,
    extract_citation_occurrences,
    generate_grounded_answer,
    score_confidence,
    verify_grounded_answer,
)
from rag_pipeline.generation.models import CitationVerificationReport, Evidence
from rag_pipeline.indexing import index_chunks
from rag_pipeline.ingestion import ingest_document
from rag_pipeline.retrieval import retrieve_reranked
from rag_pipeline.retrieval.models import RerankedRetrievalResult

from .generation.conftest import FakeCitationJudge
from .test_sample_corpus_grounded_generation import (
    _KeywordReranker,
    _make_keyword_citing_generator,
)
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


def _reranked(
    question: str, settings: Settings, provider: ConceptEmbeddingProvider, keywords: Sequence[str]
) -> list[RerankedRetrievalResult]:
    reranker = _KeywordReranker(keywords)
    return list(
        retrieve_reranked(
            question,
            ChunkingStrategy.RECURSIVE,
            settings,
            reranker,
            embedding_provider=provider,
        )
    )


def _all_verdict_judge(answer: GroundedAnswer, verdict: str) -> FakeCitationJudge:
    """A canned judge that returns `verdict` for every citation occurrence in `answer`."""
    occurrences = extract_citation_occurrences(answer.answer_text)
    return FakeCitationJudge(
        {o.occurrence_id: (o.citation_number, verdict, "canned verdict") for o in occurrences}
    )


def _evidence_from(result: RerankedRetrievalResult, *, number: int = 1) -> Evidence:
    """Build one numbered `Evidence` from a real reranked corpus chunk (real provenance)."""
    return Evidence(
        citation_number=number,
        chunk_id=result.chunk_id,
        text=result.text,
        source_file=result.source_file,
        document_id=result.document_id,
        chunk_index=result.chunk_index,
        section_heading=result.section_heading,
        page_number=result.page_number,
        chunking_strategy=result.chunking_strategy,
        reranked_rank=number,
    )


def test_fully_supported_db_answer_has_high_citation_support_component(
    pipeline_settings: Settings,
) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    question = "What does ERR_DB_1042 mean and how do I troubleshoot it?"
    results = _reranked(question, pipeline_settings, provider, ["database", "err_db_1042", "pool"])
    generator = _make_keyword_citing_generator(
        "ERR_DB_1042 indicates connection pool exhaustion.", keywords=["err_db_1042", "pool"]
    )
    answer = generate_grounded_answer(question, results, generator)
    assert answer.cited_numbers

    report = verify_grounded_answer(question, answer, _all_verdict_judge(answer, "supported"))
    assessment = score_confidence(answer, report, results, pipeline_settings)

    assert assessment.citation_support_score == pytest.approx(1.0)
    assert assessment.supported_count == assessment.total_citation_occurrences
    assert assessment.has_contradiction is False
    assert assessment.is_insufficient_evidence is False
    # citation support dominates (weight 0.9), so a fully-supported answer
    # scores high regardless of the weaker retrieval-agreement component.
    assert assessment.score >= 0.9


def test_supported_dual_channel_cited_chunk_gets_nonzero_retrieval_agreement(
    pipeline_settings: Settings,
) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    question = "What does ERR_DB_1042 mean and how do I troubleshoot it?"
    results = _reranked(question, pipeline_settings, provider, ["database", "err_db_1042", "pool"])
    dual = next(
        (r for r in results if r.dense_rank is not None and r.sparse_rank is not None), None
    )
    assert dual is not None, "expected a reranked chunk found by BOTH dense and sparse channels"

    # Cite exactly that one dual-channel chunk.
    answer = GroundedAnswer(
        answer_text="ERR_DB_1042 indicates connection pool exhaustion [1].",
        evidence=(_evidence_from(dual),),
        cited_numbers=(1,),
    )
    report = verify_grounded_answer(question, answer, _all_verdict_judge(answer, "supported"))
    assessment = score_confidence(answer, report, results, pipeline_settings)

    assert assessment.unique_cited_evidence_count == 1
    assert assessment.dual_channel_cited_evidence_count == 1
    assert assessment.retrieval_agreement_score == pytest.approx(1.0)
    # both components at 1.0 -> normalized composite at 1.0
    assert assessment.score == pytest.approx(1.0)


def test_intentionally_contradicted_token_duration_answer_exposes_has_contradiction(
    pipeline_settings: Settings,
) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    results = _reranked(
        "access token lifetime expiry minutes",
        pipeline_settings,
        provider,
        ["token", "minute", "expire", "lifetime"],
    )
    token_hit = next(r for r in results if "60 minutes" in r.text.lower())

    # Deliberately wrong: the real evidence says 60 minutes.
    answer = GroundedAnswer(
        answer_text="Access tokens expire after 24 hours [1].",
        evidence=(_evidence_from(token_hit),),
        cited_numbers=(1,),
    )
    judge = FakeCitationJudge(
        {1: (1, "contradicted", "Evidence states 60 minutes, claim states 24 hours.")}
    )
    report = verify_grounded_answer("How long do tokens last?", answer, judge)
    assessment = score_confidence(answer, report, results, pipeline_settings)

    assert assessment.has_contradiction is True
    assert assessment.contradicted_count == 1
    assert assessment.citation_support_score == pytest.approx(0.0)
    # Step 3 exposes the contradiction but does NOT cap/override the score.
    assert assessment.is_insufficient_evidence is False


def test_partially_supported_answer_lowers_citation_support_component(
    pipeline_settings: Settings,
) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    question = "What does ERR_DB_1042 mean and how do I troubleshoot it?"
    results = _reranked(question, pipeline_settings, provider, ["database", "err_db_1042", "pool"])
    generator = _make_keyword_citing_generator(
        "ERR_DB_1042 indicates connection pool exhaustion.", keywords=["err_db_1042", "pool"]
    )
    answer = generate_grounded_answer(question, results, generator)
    assert answer.cited_numbers

    supported = score_confidence(
        answer,
        verify_grounded_answer(question, answer, _all_verdict_judge(answer, "supported")),
        results,
        pipeline_settings,
    )
    partial = score_confidence(
        answer,
        verify_grounded_answer(question, answer, _all_verdict_judge(answer, "partially_supported")),
        results,
        pipeline_settings,
    )

    assert partial.citation_support_score == pytest.approx(0.5)
    assert partial.citation_support_score < supported.citation_support_score
    assert partial.score < supported.score
    assert partial.partially_supported_count == partial.total_citation_occurrences


def test_repeated_citation_does_not_inflate_retrieval_agreement(
    pipeline_settings: Settings,
) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    results = _reranked(
        "access token lifetime expiry minutes",
        pipeline_settings,
        provider,
        ["token", "minute", "expire", "lifetime"],
    )
    token_hit = next(r for r in results if "60 minutes" in r.text.lower())

    # [1] cited twice, both occurrences supported.
    answer = GroundedAnswer(
        answer_text="Access tokens expire after 60 minutes [1]. They must then be refreshed [1].",
        evidence=(_evidence_from(token_hit),),
        cited_numbers=(1,),
    )
    judge = FakeCitationJudge(
        {
            1: (1, "supported", "Evidence confirms 60-minute expiry."),
            2: (1, "supported", "Evidence implies a refresh is needed afterwards."),
        }
    )
    report = verify_grounded_answer("q", answer, judge)
    assessment = score_confidence(answer, report, results, pipeline_settings)

    assert assessment.total_citation_occurrences == 2
    assert assessment.unique_cited_evidence_count == 1
    assert assessment.dual_channel_cited_evidence_count <= 1
    assert assessment.retrieval_agreement_score <= 1.0
    assert assessment.citation_support_score == pytest.approx(1.0)


def test_office_parking_insufficient_evidence_response_scores_zero(
    pipeline_settings: Settings,
) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    question = "How many office parking spots are reserved for visitors?"
    results = _reranked(question, pipeline_settings, provider, ["office", "parking"])
    generator = _make_keyword_citing_generator(
        "Employees may park in the garage.", keywords=["office parking", "reserved spot"]
    )
    answer = generate_grounded_answer(question, results, generator)
    assert answer.cited_numbers == ()
    assert "do not provide enough information" in answer.answer_text.lower()

    judge = FakeCitationJudge(error=RuntimeError("judge must never be called"))
    report = verify_grounded_answer(question, answer, judge)
    assessment = score_confidence(answer, report, results, pipeline_settings)

    assert isinstance(report, CitationVerificationReport)
    assert assessment.is_insufficient_evidence is True
    assert assessment.score == pytest.approx(0.0)
    assert assessment.citation_support_score == pytest.approx(0.0)
    assert assessment.retrieval_agreement_score == pytest.approx(0.0)
    assert assessment.total_citation_occurrences == 0
    assert assessment.unique_cited_evidence_count == 0
    assert assessment.has_contradiction is False
    assert judge.calls == []
