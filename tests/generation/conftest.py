"""Shared fixtures/helpers for generation tests."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from rag_pipeline.config import ChunkingStrategy, Settings
from rag_pipeline.generation.base import RawJudgeVerdict
from rag_pipeline.generation.citations import extract_citations
from rag_pipeline.generation.context import build_evidence
from rag_pipeline.generation.models import GroundedAnswer
from rag_pipeline.retrieval.models import RerankedRetrievalResult


@pytest.fixture
def index_settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, index_root_dir=tmp_path / "indexes")


def make_reranked_result(
    *,
    chunk_id: str,
    rank: int,
    text: str | None = None,
    reranker_score: float = 1.0,
    hybrid_rank: int | None = None,
    rrf_score: float = 0.01,
    dense_rank: int | None = 1,
    sparse_rank: int | None = None,
    dense_contribution: float = 0.01,
    sparse_contribution: float = 0.0,
    dense_distance: float | None = 0.1,
    dense_similarity: float | None = 0.9,
    bm25_score: float | None = None,
    document_id: str = "d" * 64,
    chunk_index: int = 0,
    source_file: str = "doc.md",
    section_heading: str | None = None,
    page_number: int | None = None,
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
) -> RerankedRetrievalResult:
    return RerankedRetrievalResult(
        chunk_id=chunk_id,
        rank=rank,
        reranker_score=reranker_score,
        hybrid_rank=hybrid_rank if hybrid_rank is not None else rank,
        rrf_score=rrf_score,
        dense_rank=dense_rank,
        sparse_rank=sparse_rank,
        dense_contribution=dense_contribution,
        sparse_contribution=sparse_contribution,
        dense_distance=dense_distance,
        dense_similarity=dense_similarity,
        bm25_score=bm25_score,
        text=text if text is not None else f"text for {chunk_id}",
        document_id=document_id,
        chunk_index=chunk_index,
        source_file=source_file,
        section_heading=section_heading,
        page_number=page_number,
        chunking_strategy=chunking_strategy,
    )


class FakeGenerator:
    """Deterministic, network-free `Generator` double for offline generation tests.

    `response` is returned verbatim from every call. `response_fn`, if
    given instead, is called with `(system_prompt, user_prompt)` to
    compute a response dynamically -- e.g. to react to which evidence
    numbers/content actually appear in the rendered prompt, which real
    retrieval output doesn't let a test predict in advance. `error`, if
    given, is raised instead of generating at all (simulating a provider
    failure). Every call's exact prompts are recorded in `.calls`.
    """

    def __init__(
        self,
        response: str | None = None,
        *,
        response_fn: Callable[[str, str], str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._response = response
        self._response_fn = response_fn
        self._error = error
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if self._error is not None:
            raise self._error
        if self._response_fn is not None:
            return self._response_fn(system_prompt, user_prompt)
        assert self._response is not None
        return self._response


def make_grounded_answer(
    *,
    answer_text: str,
    reranked_results: Sequence[RerankedRetrievalResult] | None = None,
) -> GroundedAnswer:
    """Build a `GroundedAnswer` directly, bypassing `generate_grounded_answer()`.

    Lets verification tests control `answer_text`/evidence precisely
    (including deliberately-contradictory or malformed text) without
    needing a `Generator` round-trip -- `GroundedAnswer` has no
    construction-time validation of its own (validation lives in
    `generate_grounded_answer()`), so this is a legitimate, direct way
    to build test fixtures.
    """
    results = (
        reranked_results
        if reranked_results is not None
        else [make_reranked_result(chunk_id="a", rank=1)]
    )
    evidence = build_evidence(results)
    cited_numbers = extract_citations(answer_text)
    return GroundedAnswer(
        answer_text=answer_text,
        evidence=tuple(evidence),
        cited_numbers=tuple(cited_numbers),
    )


class FakeCitationJudge:
    """Deterministic, network-free `CitationJudge` double for offline verification tests.

    `verdicts_by_occurrence` maps `occurrence_id -> (citation_number,
    verdict, rationale)` to return for that occurrence -- a well-formed
    response covering exactly the given occurrence IDs. `override_raw`,
    if given, is returned verbatim instead (as a list of
    `RawJudgeVerdict`) -- lets tests simulate malformed judge output
    (missing/duplicate/extra occurrence, wrong citation number, invalid
    verdict, empty rationale) freely. `error`, if given, is raised
    instead of judging at all (simulating a provider failure). Every
    call's exact prompts are recorded in `.calls`.
    """

    def __init__(
        self,
        verdicts_by_occurrence: dict[int, tuple[int, str, str]] | None = None,
        *,
        override_raw: list[RawJudgeVerdict] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._verdicts_by_occurrence = verdicts_by_occurrence or {}
        self._override_raw = override_raw
        self._error = error
        self.calls: list[tuple[str, str]] = []

    def judge(self, system_prompt: str, user_prompt: str) -> list[RawJudgeVerdict]:
        self.calls.append((system_prompt, user_prompt))
        if self._error is not None:
            raise self._error
        if self._override_raw is not None:
            return self._override_raw
        return [
            RawJudgeVerdict(
                occurrence_id=occurrence_id,
                citation_number=citation_number,
                verdict=verdict,
                rationale=rationale,
            )
            for occurrence_id, (
                citation_number,
                verdict,
                rationale,
            ) in self._verdicts_by_occurrence.items()
        ]
