"""Tests for citation *occurrence* extraction and judge-only answer annotation."""

from __future__ import annotations

from rag_pipeline.generation.citations import extract_citation_occurrences
from rag_pipeline.generation.judge_prompt import annotate_answer
from rag_pipeline.generation.models import CitationOccurrence


def test_one_citation_creates_one_occurrence() -> None:
    occurrences = extract_citation_occurrences("Tokens expire in 60 minutes [1].")
    assert len(occurrences) == 1
    assert occurrences[0].occurrence_id == 1
    assert occurrences[0].citation_number == 1


def test_repeated_same_citation_creates_multiple_occurrences() -> None:
    occurrences = extract_citation_occurrences("A [1]. B [2]. C [1].")
    assert [o.citation_number for o in occurrences] == [1, 2, 1]
    assert [o.occurrence_id for o in occurrences] == [1, 2, 3]


def test_adjacent_citations_create_two_occurrences() -> None:
    occurrences = extract_citation_occurrences("Supported by two sources [1][2].")
    assert [o.citation_number for o in occurrences] == [1, 2]
    assert [o.occurrence_id for o in occurrences] == [1, 2]


def test_spaced_citations_create_two_occurrences() -> None:
    occurrences = extract_citation_occurrences("Supported by two sources [1] [2].")
    assert [o.citation_number for o in occurrences] == [1, 2]
    assert [o.occurrence_id for o in occurrences] == [1, 2]


def test_occurrence_ids_are_appearance_ordered_starting_at_one() -> None:
    occurrences = extract_citation_occurrences("A [3]. B [1]. C [2].")
    assert [o.occurrence_id for o in occurrences] == [1, 2, 3]
    assert [o.citation_number for o in occurrences] == [3, 1, 2]


def test_offsets_point_to_the_correct_citation_text() -> None:
    text = "Tokens expire after 60 minutes [1]."
    occurrences = extract_citation_occurrences(text)
    occurrence = occurrences[0]
    assert text[occurrence.start_offset : occurrence.end_offset] == "[1]"


def test_no_citations_returns_empty_list() -> None:
    assert extract_citation_occurrences("No citations here.") == []


def test_non_digit_bracket_does_not_create_an_occurrence() -> None:
    occurrences = extract_citation_occurrences("See [note] and [Fig. 1], not real citations.")
    assert occurrences == []


# --- judge-only annotation never touches the user-facing answer -------------------


def test_annotate_answer_wraps_each_occurrence_with_a_marker() -> None:
    text = "A [1]. B [2]."
    occurrences = extract_citation_occurrences(text)
    annotated = annotate_answer(text, occurrences)
    assert '<occurrence id="1">[1]</occurrence>' in annotated
    assert '<occurrence id="2">[2]</occurrence>' in annotated


def test_annotate_answer_preserves_surrounding_text() -> None:
    text = "Access tokens expire after 60 minutes [1]."
    occurrences = extract_citation_occurrences(text)
    annotated = annotate_answer(text, occurrences)
    assert annotated.startswith("Access tokens expire after 60 minutes ")
    assert annotated.endswith(".")


def test_annotate_answer_does_not_mutate_or_return_the_original_string() -> None:
    text = "A [1]. B [2]. C [1]."
    occurrences = extract_citation_occurrences(text)
    annotated = annotate_answer(text, occurrences)
    assert annotated != text
    assert text == "A [1]. B [2]. C [1]."  # the original object/content is untouched


def test_annotate_answer_distinguishes_repeated_citation_occurrences() -> None:
    text = "A [1]. B [2]. C [1]."
    occurrences = extract_citation_occurrences(text)
    annotated = annotate_answer(text, occurrences)
    # occurrence 1 and occurrence 3 both cite [1], but get distinct markers.
    assert '<occurrence id="1">[1]</occurrence>' in annotated
    assert '<occurrence id="3">[1]</occurrence>' in annotated
    assert annotated.count("[1]</occurrence>") == 2


def test_annotate_answer_with_zero_occurrences_returns_equivalent_text() -> None:
    text = "No citations here at all."
    annotated = annotate_answer(text, [])
    assert annotated == text


def test_annotate_answer_handles_out_of_order_occurrence_list() -> None:
    # extract_citation_occurrences always returns appearance order, but
    # annotate_answer must not assume its input list is sorted.
    text = "A [1]. B [2]."
    occurrences = extract_citation_occurrences(text)
    reversed_occurrences = list(reversed(occurrences))
    annotated = annotate_answer(text, reversed_occurrences)
    assert '<occurrence id="1">[1]</occurrence>' in annotated
    assert '<occurrence id="2">[2]</occurrence>' in annotated


def test_citation_occurrence_is_frozen() -> None:
    occurrence = CitationOccurrence(
        occurrence_id=1, citation_number=1, start_offset=0, end_offset=3
    )
    import dataclasses

    assert dataclasses.is_dataclass(occurrence)
    try:
        occurrence.occurrence_id = 2  # type: ignore[misc]
        raised = False
    except dataclasses.FrozenInstanceError:
        raised = True
    assert raised
