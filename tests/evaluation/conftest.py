"""Shared fixtures/helpers for evaluation tests (offline, no pipeline, no network)."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from rag_pipeline.config import ChunkingStrategy
from rag_pipeline.evaluation.metrics.correctness import (
    RawCorrectnessAssessment,
    RawFactVerdict,
    RawGoldenContradiction,
)
from rag_pipeline.evaluation.metrics.faithfulness import RawClaimVerdict
from rag_pipeline.evaluation.models import (
    Answerability,
    Difficulty,
    GoldenQACase,
    QuestionType,
)
from rag_pipeline.generation.citations import extract_citation_occurrences, extract_citations
from rag_pipeline.generation.models import (
    AnswerDecision,
    CitationVerdict,
    CitationVerification,
    CitationVerificationReport,
    ConfidenceAssessment,
    Evidence,
    FinalAnswer,
    GroundedAnswer,
)
from rag_pipeline.retrieval.models import DenseRetrievalResult, SparseRetrievalResult

Record = Callable[..., dict[str, object]]


# --- golden dataset record fixtures (Step 1 parser/validation tests) -------------


def _answerable_record(**overrides: object) -> dict[str, object]:
    """A minimal valid ANSWERABLE case: single-source, non-multi, unconstrained category."""
    record: dict[str, object] = {
        "id": "case-1",
        "question": "How often are full PostgreSQL backups taken?",
        "answerability": "answerable",
        "question_type": "direct_factual",
        "difficulty": "easy",
        "requires_multi_document_reasoning": False,
        "expected_answer": "Full PostgreSQL backups run daily.",
        "expected_facts": ["Full PostgreSQL backups run daily"],
        "expected_source_files": ["database-operations.md"],
        "expected_identifiers": [],
    }
    record.update(overrides)
    return record


def _exact_identifier_record(**overrides: object) -> dict[str, object]:
    """A minimal valid ANSWERABLE `exact_identifier` case (identifier present in the question)."""
    record: dict[str, object] = {
        "id": "exact-1",
        "question": "What does ERR_DB_1042 mean?",
        "answerability": "answerable",
        "question_type": "exact_identifier",
        "difficulty": "easy",
        "requires_multi_document_reasoning": False,
        "expected_answer": "A database connection could not be obtained from the pool.",
        "expected_facts": ["ERR_DB_1042 means a database connection could not be obtained"],
        "expected_source_files": ["api-error-codes.txt"],
        "expected_identifiers": ["ERR_DB_1042"],
    }
    record.update(overrides)
    return record


def _unanswerable_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "id": "absent-1",
        "question": "How many visitor parking spaces does the office have?",
        "answerability": "unanswerable",
        "question_type": "unanswerable_absent",
        "difficulty": "easy",
        "requires_multi_document_reasoning": False,
        "expected_answer": None,
        "notes": "The corpus never mentions an office or parking.",
    }
    record.update(overrides)
    return record


@pytest.fixture
def answerable_record() -> Record:
    return _answerable_record


@pytest.fixture
def exact_identifier_record() -> Record:
    return _exact_identifier_record


@pytest.fixture
def unanswerable_record() -> Record:
    return _unanswerable_record


@pytest.fixture
def write_jsonl(tmp_path: Path) -> Callable[[list[dict[str, object]]], Path]:
    def _write(records: list[dict[str, object]]) -> Path:
        path = tmp_path / "golden_qa.jsonl"
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
        )
        return path

    return _write


# --- Step 2 metric builders ----------------------------------------------------
#
# These construct `GoldenQACase` / `FinalAnswer` and friends DIRECTLY, bypassing
# `parse_golden_case()` and the production pipeline. The frozen models have no
# construction-time validation (that lives in the parse/service functions), so
# this is a legitimate way to build precise, deliberately-varied test fixtures.


def make_golden_case(**overrides: object) -> GoldenQACase:
    base: dict[str, object] = {
        "id": "case-1",
        "question": "What is the access token lifetime?",
        "answerability": Answerability.ANSWERABLE,
        "question_type": QuestionType.DIRECT_FACTUAL,
        "difficulty": Difficulty.EASY,
        "requires_multi_document_reasoning": False,
        "expected_answer": "Access tokens live for 60 minutes.",
        "expected_facts": ("Access tokens expire after 60 minutes",),
        "expected_source_files": ("authentication-api.md",),
        "expected_identifiers": (),
        "acceptable_source_files": (),
        "tags": (),
        "notes": None,
    }
    base.update(overrides)
    return GoldenQACase(**base)  # type: ignore[arg-type]


def make_chunk(
    *,
    source_file: str,
    text: str = "",
    chunk_id: str | None = None,
    similarity: float = 0.9,
    bm25_score: float = 1.0,
    sparse: bool = False,
) -> DenseRetrievalResult | SparseRetrievalResult:
    """A retrieval result carrying a native score, for retrieval-metric tests.

    `sparse=False` -> `DenseRetrievalResult` (has `similarity`/`distance`);
    `sparse=True` -> `SparseRetrievalResult` (has `bm25_score`). The
    metric must ignore both.
    """
    cid = chunk_id if chunk_id is not None else f"chunk-{source_file}-{abs(hash(text)) % 9999}"
    common = {
        "chunk_id": cid,
        "rank": 1,
        "text": text,
        "document_id": "d" * 64,
        "chunk_index": 0,
        "source_file": source_file,
        "section_heading": None,
        "page_number": None,
        "chunking_strategy": ChunkingStrategy.RECURSIVE,
    }
    if sparse:
        return SparseRetrievalResult(bm25_score=bm25_score, **common)  # type: ignore[arg-type]
    return DenseRetrievalResult(distance=1.0 - similarity, similarity=similarity, **common)  # type: ignore[arg-type]


def make_evidence(sources: Sequence[str]) -> tuple[Evidence, ...]:
    return tuple(
        Evidence(
            citation_number=i,
            chunk_id=f"chunk-{i}",
            text=f"evidence text {i}",
            source_file=src,
            document_id="doc-" + str(i),
            chunk_index=i - 1,
            section_heading=None,
            page_number=None,
            chunking_strategy=ChunkingStrategy.RECURSIVE,
            reranked_rank=i,
        )
        for i, src in enumerate(sources, start=1)
    )


def make_grounded_answer(*, answer_text: str, sources: Sequence[str]) -> GroundedAnswer:
    return GroundedAnswer(
        answer_text=answer_text,
        evidence=make_evidence(sources),
        cited_numbers=tuple(extract_citations(answer_text)),
    )


def make_verification_report(
    grounded: GroundedAnswer, verdicts: Sequence[CitationVerdict]
) -> CitationVerificationReport:
    occurrences = tuple(extract_citation_occurrences(grounded.answer_text))
    if len(occurrences) != len(verdicts):
        raise AssertionError("one verdict per citation occurrence is required")
    verifications = tuple(
        CitationVerification(
            occurrence_id=occ.occurrence_id,
            citation_number=occ.citation_number,
            verdict=verdict,
            rationale="rationale",
            chunk_id=grounded.evidence[occ.citation_number - 1].chunk_id,
        )
        for occ, verdict in zip(occurrences, verdicts, strict=True)
    )
    return CitationVerificationReport(
        grounded_answer=grounded, occurrences=occurrences, verifications=verifications
    )


def make_confidence(**overrides: object) -> ConfidenceAssessment:
    base: dict[str, object] = {
        "score": 1.0,
        "citation_support_score": 1.0,
        "retrieval_agreement_score": 1.0,
        "supported_count": 0,
        "partially_supported_count": 0,
        "unsupported_count": 0,
        "contradicted_count": 0,
        "total_citation_occurrences": 0,
        "unique_cited_evidence_count": 0,
        "dual_channel_cited_evidence_count": 0,
        "has_contradiction": False,
        "is_insufficient_evidence": False,
        "citation_weight": 0.9,
        "retrieval_agreement_weight": 0.1,
    }
    base.update(overrides)
    return ConfidenceAssessment(**base)  # type: ignore[arg-type]


def make_final_answer(
    *,
    decision: AnswerDecision,
    grounded: GroundedAnswer,
    report: CitationVerificationReport | None = None,
    confidence: ConfidenceAssessment | None = None,
    abstention_reason: str | None = None,
) -> FinalAnswer:
    abstained = decision is not AnswerDecision.ANSWERED
    answer_text = "I don't know." if abstained else grounded.answer_text
    if report is None:
        report = CitationVerificationReport(
            grounded_answer=grounded, occurrences=(), verifications=()
        )
    return FinalAnswer(
        answer_text=answer_text,
        decision=decision,
        grounded_answer=grounded,
        verification_report=report,
        confidence=confidence if confidence is not None else make_confidence(),
        abstained=abstained,
        abstention_reason=abstention_reason if abstained else None,
    )


# --- fake semantic judges (offline) ------------------------------------------


class FakeCorrectnessJudge:
    """Deterministic `CorrectnessJudge` double.

    `verdicts` -> list of `(fact_id, verdict_str, rationale)` for OUTPUT A.
    `contradictions` -> list of `(contradiction_id, claim_text, rationale)`
    or `(contradiction_id, claim_text, rationale, conflicting_fact_ids)`
    for OUTPUT B (defaults to an empty list -- the normal case).
    `raw` / `raw_contradictions` inject a ready list of `RawFactVerdict` /
    `RawGoldenContradiction` verbatim, for malformed-output tests. `error`
    is raised instead of judging. Prompts are recorded in `.calls`.
    """

    def __init__(
        self,
        verdicts: Sequence[tuple[int, str, str]] | None = None,
        *,
        contradictions: Sequence[tuple[int, str, str] | tuple[int, str, str, Sequence[int]]]
        | None = None,
        raw: list[RawFactVerdict] | None = None,
        raw_contradictions: list[RawGoldenContradiction] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._verdicts = verdicts
        self._contradictions = contradictions
        self._raw = raw
        self._raw_contradictions = raw_contradictions
        self._error = error
        self.calls: list[tuple[str, str]] = []

    def assess_correctness(self, system_prompt: str, user_prompt: str) -> RawCorrectnessAssessment:
        self.calls.append((system_prompt, user_prompt))
        if self._error is not None:
            raise self._error
        fact_verdicts = (
            self._raw
            if self._raw is not None
            else [
                RawFactVerdict(fact_id=fid, verdict=verdict, rationale=rationale)
                for fid, verdict, rationale in (self._verdicts or [])
            ]
        )
        if self._raw_contradictions is not None:
            contradictions = self._raw_contradictions
        else:
            contradictions = [
                RawGoldenContradiction(
                    contradiction_id=item[0],
                    claim_text=item[1],
                    rationale=item[2],
                    conflicting_fact_ids=tuple(item[3]) if len(item) > 3 else (),
                )
                for item in (self._contradictions or [])
            ]
        return RawCorrectnessAssessment(
            fact_verdicts=fact_verdicts, golden_contradictions=contradictions
        )


class FakeFaithfulnessJudge:
    """Deterministic `FaithfulnessJudge` double (see `FakeCorrectnessJudge`)."""

    def __init__(
        self,
        claims: Sequence[tuple[int, str, str, str]] | None = None,
        *,
        raw: list[RawClaimVerdict] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._claims = claims
        self._raw = raw
        self._error = error
        self.calls: list[tuple[str, str]] = []

    def assess_faithfulness(self, system_prompt: str, user_prompt: str) -> list[RawClaimVerdict]:
        self.calls.append((system_prompt, user_prompt))
        if self._error is not None:
            raise self._error
        if self._raw is not None:
            return self._raw
        return [
            RawClaimVerdict(claim_id=cid, claim_text=text, verdict=verdict, rationale=rationale)
            for cid, text, verdict, rationale in (self._claims or [])
        ]
