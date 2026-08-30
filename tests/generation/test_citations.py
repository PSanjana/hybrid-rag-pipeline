"""Tests for rag_pipeline.generation.citations (deterministic bracket-citation parsing)."""

from __future__ import annotations

import pytest

from rag_pipeline.config import ChunkingStrategy
from rag_pipeline.generation.citations import (
    extract_citations,
    resolve_citation,
    validate_citations,
    validate_evidence_numbering,
)
from rag_pipeline.generation.context import build_evidence
from rag_pipeline.generation.exceptions import CitationValidationError
from rag_pipeline.generation.models import Evidence

from .conftest import make_reranked_result


def _make_evidence(citation_number: object, chunk_id: str = "a") -> Evidence:
    return Evidence(
        citation_number=citation_number,  # type: ignore[arg-type]
        chunk_id=chunk_id,
        text=f"text for {chunk_id}",
        source_file="doc.md",
        document_id="d" * 64,
        chunk_index=0,
        section_heading=None,
        page_number=None,
        chunking_strategy=ChunkingStrategy.RECURSIVE,
        reranked_rank=1,
    )


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


# --- evidence-numbering validation --------------------------------------------------


def test_validate_evidence_numbering_accepts_well_formed_sequence() -> None:
    evidence = (_make_evidence(1, "a"), _make_evidence(2, "b"), _make_evidence(3, "c"))
    by_number = validate_evidence_numbering(evidence)
    assert by_number[1].chunk_id == "a"
    assert by_number[2].chunk_id == "b"
    assert by_number[3].chunk_id == "c"


def test_validate_evidence_numbering_accepts_empty_sequence() -> None:
    assert validate_evidence_numbering(()) == {}


def test_validate_evidence_numbering_rejects_duplicate_citation_numbers() -> None:
    evidence = (_make_evidence(1, "a"), _make_evidence(1, "b"))
    with pytest.raises(CitationValidationError):
        validate_evidence_numbering(evidence)


def test_validate_evidence_numbering_rejects_gap() -> None:
    evidence = (_make_evidence(1, "a"), _make_evidence(3, "b"))
    with pytest.raises(CitationValidationError):
        validate_evidence_numbering(evidence)


def test_validate_evidence_numbering_rejects_missing_leading_number() -> None:
    evidence = (_make_evidence(2, "a"), _make_evidence(3, "b"))
    with pytest.raises(CitationValidationError):
        validate_evidence_numbering(evidence)


def test_validate_evidence_numbering_rejects_wrong_order() -> None:
    evidence = (_make_evidence(2, "a"), _make_evidence(1, "b"))
    with pytest.raises(CitationValidationError):
        validate_evidence_numbering(evidence)


def test_validate_evidence_numbering_rejects_zero() -> None:
    evidence = (_make_evidence(0, "a"),)
    with pytest.raises(CitationValidationError):
        validate_evidence_numbering(evidence)


def test_validate_evidence_numbering_rejects_negative() -> None:
    evidence = (_make_evidence(-1, "a"),)
    with pytest.raises(CitationValidationError):
        validate_evidence_numbering(evidence)


def test_validate_evidence_numbering_rejects_non_integer() -> None:
    evidence = (_make_evidence("1", "a"),)
    with pytest.raises(CitationValidationError):
        validate_evidence_numbering(evidence)


def test_validate_evidence_numbering_rejects_bool_masquerading_as_int() -> None:
    evidence = (_make_evidence(True, "a"),)
    with pytest.raises(CitationValidationError):
        validate_evidence_numbering(evidence)


def test_resolve_citation_rejects_duplicate_citation_numbers_without_leaking_key_error() -> None:
    evidence = (_make_evidence(1, "a"), _make_evidence(1, "b"))
    with pytest.raises(CitationValidationError):
        resolve_citation(evidence, 1)


def test_resolve_citation_rejects_malformed_evidence_ordering() -> None:
    evidence = (_make_evidence(2, "a"), _make_evidence(1, "b"))
    with pytest.raises(CitationValidationError):
        resolve_citation(evidence, 1)
