"""Tests for rag_pipeline.generation.citations (deterministic bracket-citation parsing)."""

from __future__ import annotations

import pytest

from rag_pipeline.generation.citations import (
    extract_citations,
    resolve_citation,
    validate_citations,
)
from rag_pipeline.generation.context import build_evidence
from rag_pipeline.generation.exceptions import CitationValidationError

from .conftest import make_reranked_result


def test_single_citation_extracted() -> None:
    assert extract_citations("Tokens expire in 60 minutes [1].") == [1]


def test_multiple_citations_extracted() -> None:
    assert extract_citations("A [1]. B [2]. C [3].") == [1, 2, 3]


def test_adjacent_citations_extracted() -> None:
    assert extract_citations("Supported by two sources [1][2].") == [1, 2]


def test_space_separated_adjacent_citations_extracted() -> None:
    assert extract_citations("Supported by two sources [1] [2].") == [1, 2]


def test_repeated_citation_deduplicated_at_first_appearance() -> None:
    assert extract_citations("A [2]. B [1]. C [2] again.") == [2, 1]


def test_no_citations_returns_empty_list() -> None:
    assert extract_citations("No citations here at all.") == []


def test_bracket_with_non_digit_content_is_not_a_citation() -> None:
    assert extract_citations("See the [note] below, not [Fig. 1] either.") == []


def test_negative_bracket_syntax_is_not_recognized_as_a_citation() -> None:
    assert extract_citations("Weird syntax [-1] is not a citation.") == []


# --- range validation ----------------------------------------------------------------


def test_valid_upper_boundary_accepted() -> None:
    validate_citations([1, 2, 3], evidence_count=3)  # must not raise


def test_zero_citation_rejected() -> None:
    with pytest.raises(CitationValidationError):
        validate_citations([0], evidence_count=3)


def test_out_of_range_above_count_rejected() -> None:
    with pytest.raises(CitationValidationError):
        validate_citations([4], evidence_count=3)


def test_mixed_valid_and_invalid_citations_rejected() -> None:
    with pytest.raises(CitationValidationError, match=r"\[4\]") as exc_info:
        validate_citations([1, 4], evidence_count=3)
    assert "4" in str(exc_info.value)


def test_empty_citations_with_zero_evidence_does_not_raise() -> None:
    validate_citations([], evidence_count=0)  # must not raise


# --- resolution ------------------------------------------------------------------


def test_resolve_citation_returns_matching_evidence_provenance() -> None:
    results = [
        make_reranked_result(chunk_id="a", rank=1, source_file="auth.md"),
        make_reranked_result(chunk_id="b", rank=2, source_file="access.md"),
    ]
    evidence = build_evidence(results)
    resolved = resolve_citation(evidence, 2)
    assert resolved.source_file == "access.md"
    assert resolved.chunk_id == "b"


def test_resolve_citation_rejects_out_of_range_number() -> None:
    results = [make_reranked_result(chunk_id="a", rank=1)]
    evidence = build_evidence(results)
    with pytest.raises(CitationValidationError):
        resolve_citation(evidence, 2)
