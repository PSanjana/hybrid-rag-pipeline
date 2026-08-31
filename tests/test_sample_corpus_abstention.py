"""Graceful abstention policy over the synthetic Acme Cloud sample corpus (offline).

data/sample -> ingest -> recursive chunking -> indexing -> retrieve_reranked
(real hybrid + RRF + reranking, deterministic fake reranker) ->
generate_grounded_answer (deterministic keyword-citing fake generator) ->
verify_grounded_answer (deterministic fake judge) -> score_confidence ->
apply_abstention_policy.

No network/OpenAI calls, no real cross-encoder or LLM. The judge verdicts
are controlled fakes, so the decisions here follow from the fixed
verdict/threshold inputs -- they are not evidence that any real answer is
or isn't correct.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from rag_pipeline.chunking import ChunkingStrategy, chunk_document
from rag_pipeline.config import Settings
from rag_pipeline.generation import (
    ABSTENTION_TEXT,
    AnswerDecision,
    GroundedAnswer,
    apply_abstention_policy,
    extract_citation_occurrences,
    generate_grounded_answer,
    score_confidence,
    verify_grounded_answer,
)
from rag_pipeline.generation.models import Evidence
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
    return list(
        retrieve_reranked(
            question,
            ChunkingStrategy.RECURSIVE,
            settings,
            _KeywordReranker(keywords),
            embedding_provider=provider,
        )
    )


def _all_verdict_judge(answer: GroundedAnswer, verdict: str) -> FakeCitationJudge:
    occurrences = extract_citation_occurrences(answer.answer_text)
    return FakeCitationJudge(
        {o.occurrence_id: (o.citation_number, verdict, "canned verdict") for o in occurrences}
    )


def _evidence_from(result: RerankedRetrievalResult, *, number: int = 1) -> Evidence:
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


def _final(
    answer: GroundedAnswer,
    results: list[RerankedRetrievalResult],
    judge: FakeCitationJudge,
    settings: Settings,
):
    report = verify_grounded_answer("q", answer, judge)
    confidence = score_confidence(answer, report, results, settings)
    return apply_abstention_policy(answer, report, confidence, settings)


def test_supported_db_answer_returns_the_substantive_answer(
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

    final = _final(answer, results, _all_verdict_judge(answer, "supported"), pipeline_settings)
    assert final.decision is AnswerDecision.ANSWERED
    assert final.abstained is False
    assert final.answer_text == answer.answer_text


def test_supported_authentication_answer_returns_the_substantive_answer(
    pipeline_settings: Settings,
) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    question = "How long do access tokens last, and is MFA required for production access?"
    results = _reranked(question, pipeline_settings, provider, ["token", "mfa", "authentication"])
    generator = _make_keyword_citing_generator(
        "Access tokens have a limited lifetime and production access requires MFA.",
        keywords=["token", "mfa"],
    )
    answer = generate_grounded_answer(question, results, generator)
    assert answer.cited_numbers

    final = _final(answer, results, _all_verdict_judge(answer, "supported"), pipeline_settings)
    assert final.decision is AnswerDecision.ANSWERED
    assert final.answer_text == answer.answer_text


def test_office_parking_question_abstains_gracefully_with_insufficient_evidence(
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

    judge = FakeCitationJudge(error=RuntimeError("judge must never be called"))
    final = _final(answer, results, judge, pipeline_settings)
    assert final.decision is AnswerDecision.ABSTAINED_INSUFFICIENT_EVIDENCE
    assert final.answer_text == ABSTENTION_TEXT
    assert final.grounded_answer is answer  # draft retained internally


def test_contradicted_24_hour_token_claim_abstains_due_to_contradiction(
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
    answer = GroundedAnswer(
        answer_text="Access tokens expire after 24 hours [1].",
        evidence=(_evidence_from(token_hit),),
        cited_numbers=(1,),
    )
    judge = FakeCitationJudge(
        {1: (1, "contradicted", "Evidence states 60 minutes, claim states 24 hours.")}
    )
    final = _final(answer, results, judge, pipeline_settings)
    assert final.decision is AnswerDecision.ABSTAINED_CONTRADICTION
    assert final.answer_text == ABSTENTION_TEXT
    assert "24 hours" not in final.answer_text


def test_unsupported_claim_abstains_due_to_unsupported_citation(
    pipeline_settings: Settings,
) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    question = "What does ERR_DB_1042 mean and how do I troubleshoot it?"
    results = _reranked(question, pipeline_settings, provider, ["database", "err_db_1042", "pool"])
    generator = _make_keyword_citing_generator(
        "ERR_DB_1042 grants permanent database administrator access.",
        keywords=["err_db_1042", "pool"],
    )
    answer = generate_grounded_answer(question, results, generator)
    assert answer.cited_numbers

    final = _final(answer, results, _all_verdict_judge(answer, "unsupported"), pipeline_settings)
    assert final.decision is AnswerDecision.ABSTAINED_UNSUPPORTED_CITATION
    assert final.answer_text == ABSTENTION_TEXT


def test_partially_supported_answer_above_threshold_remains_answerable(
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

    lenient = Settings(
        _env_file=None,
        raw_data_dir=pipeline_settings.raw_data_dir,
        processed_data_dir=pipeline_settings.processed_data_dir,
        index_root_dir=pipeline_settings.index_root_dir,
        confidence_threshold=0.4,
    )
    final = _final(answer, results, _all_verdict_judge(answer, "partially_supported"), lenient)
    assert final.confidence.partially_supported_count >= 1
    assert final.confidence.has_contradiction is False
    assert final.confidence.unsupported_count == 0
    assert final.decision is AnswerDecision.ANSWERED


def test_low_confidence_non_contradicted_answer_abstains_due_to_threshold(
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

    # Default threshold 0.8; all-partial verdicts keep the score well below it.
    final = _final(
        answer, results, _all_verdict_judge(answer, "partially_supported"), pipeline_settings
    )
    assert final.confidence.score < pipeline_settings.confidence_threshold
    assert final.confidence.has_contradiction is False
    assert final.confidence.unsupported_count == 0
    assert final.decision is AnswerDecision.ABSTAINED_LOW_CONFIDENCE
    assert final.answer_text == ABSTENTION_TEXT
