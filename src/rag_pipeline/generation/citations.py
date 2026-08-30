"""Deterministic bracket-citation extraction, range validation, and provenance resolution.

No LLM is ever used to parse or judge citations here -- extraction is a
fixed regular expression, and validation is a plain range check.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .exceptions import CitationValidationError
from .models import CitationOccurrence, Evidence

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def extract_citations(text: str) -> list[int]:
    """Extract bracket citation numbers from `text`, in first-appearance order, deduplicated.

    Only a bracket whose *entire* content is digits (`[1]`, `[42]`) is
    treated as a citation -- adjacent/spaced multi-citations (`[1][3]`,
    `[1] [3]`) are each matched independently, since the pattern doesn't
    care what precedes or follows a bracket. A bracket containing
    anything else (`[note]`, `[Fig. 1]`, `[-1]`) is not a digit-only
    match and is therefore never extracted at all -- ordinary bracketed
    prose is left alone rather than partially/incorrectly parsed.
    Repeated citations (`[1] ... [1]`) appear once, at their first
    position, giving one deterministic canonical ordering per answer.
    """
    seen: list[int] = []
    seen_set: set[int] = set()
    for match in _CITATION_PATTERN.finditer(text):
        number = int(match.group(1))
        if number not in seen_set:
            seen_set.add(number)
            seen.append(number)
    return seen


def extract_citation_occurrences(text: str) -> list[CitationOccurrence]:
    """Extract every bracket-citation *occurrence* from `text`, in left-to-right appearance order.

    Unlike `extract_citations()` (which deduplicates to the set of
    unique cited numbers), this returns one `CitationOccurrence` per
    bracket *appearance*: `"A [1]. B [2]. C [1]."` yields three
    occurrences (citation numbers 1, 2, 1 respectively), not two.
    `occurrence_id` is 1-based and assigned strictly in appearance
    order; `start_offset`/`end_offset` delimit exactly the bracket
    substring itself, i.e. `text[start_offset:end_offset]` reproduces
    e.g. `"[1]"`. Uses the same digit-only bracket pattern as
    `extract_citations()`, so the two functions always agree on what
    counts as a citation.
    """
    occurrences: list[CitationOccurrence] = []
    for occurrence_id, match in enumerate(_CITATION_PATTERN.finditer(text), start=1):
        occurrences.append(
            CitationOccurrence(
                occurrence_id=occurrence_id,
                citation_number=int(match.group(1)),
                start_offset=match.start(),
                end_offset=match.end(),
            )
        )
    return occurrences


def validate_citations(cited_numbers: Sequence[int], evidence_count: int) -> None:
    """Every cited number must fall within `[1, evidence_count]`.

    With `evidence_count` evidence blocks supplied, valid references are
    exactly `[1]` through `[evidence_count]`; `[0]`, any number above
    `evidence_count`, or (were the parser ever to recognize such syntax)
    a negative number are all rejected. All out-of-range numbers are
    named in the raised error, not just the first, and invalid
    citations are never silently dropped/repaired.
    """
    invalid = sorted({n for n in cited_numbers if n < 1 or n > evidence_count})
    if invalid:
        raise CitationValidationError(
            f"Generated answer cites out-of-range citation number(s) {invalid}; valid range "
            f"is [1, {evidence_count}]."
        )


def validate_evidence_numbering(evidence: Sequence[Evidence]) -> dict[int, Evidence]:
    """Validate that `evidence`'s citation numbers are exactly `1..len(evidence)` in order.

    A single positional check -- `evidence[i].citation_number` must
    equal `i + 1` exactly -- subsumes duplicate, missing, gapped, and
    out-of-order numbering all at once: if every position's number
    strictly equals its expected 1-based position, no two positions can
    share a number and no number can be skipped. `bool` is rejected
    explicitly and checked *before* the equality comparison, since
    Python's `bool` is an `int` subclass and `True == 1` -- a stray
    `True` would otherwise silently pass position 1's check. Never
    builds the `citation_number -> Evidence` mapping via a bare dict
    comprehension, which would let a duplicate/malformed
    `citation_number` silently overwrite an earlier `Evidence` instead
    of being rejected. Shared by `resolve_citation()` and
    `generation.verification` so both enforce the identical invariant
    and never raise a raw `KeyError` from malformed evidence.
    """
    numbers: dict[int, Evidence] = {}
    for index, item in enumerate(evidence, start=1):
        number = item.citation_number
        if isinstance(number, bool) or not isinstance(number, int):
            raise CitationValidationError(
                f"Evidence item at position {index} has a non-integer citation_number: {number!r}."
            )
        if number != index:
            raise CitationValidationError(
                f"Evidence citation numbers must be exactly 1..{len(evidence)} in order; "
                f"evidence item at position {index} has citation_number={number!r}."
            )
        numbers[number] = item
    return numbers


def resolve_citation(evidence: Sequence[Evidence], citation_number: int) -> Evidence:
    """Resolve one citation number back to its `Evidence` (source_file, section, page, chunk_id).

    Raises `CitationValidationError` if `evidence`'s own citation
    numbering is malformed (see `validate_evidence_numbering`), or if
    `citation_number` is outside the supplied evidence's range -- the
    same failure mode as `validate_citations`, since both express the
    same "citation must refer to supplied evidence" invariant. Never
    leaks a raw `KeyError`.
    """
    by_number = validate_evidence_numbering(evidence)
    if citation_number not in by_number:
        raise CitationValidationError(
            f"Citation number {citation_number!r} does not refer to any of the "
            f"{len(evidence)} supplied evidence item(s)."
        )
    return by_number[citation_number]
