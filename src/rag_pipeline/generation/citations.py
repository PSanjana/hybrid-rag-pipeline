"""Deterministic bracket-citation extraction, range validation, and provenance resolution.

No LLM is ever used to parse or judge citations here -- extraction is a
fixed regular expression, and validation is a plain range check.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .exceptions import CitationValidationError
from .models import Evidence

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


def resolve_citation(evidence: Sequence[Evidence], citation_number: int) -> Evidence:
    """Resolve one citation number back to its `Evidence` (source_file, section, page, chunk_id).

    Raises `CitationValidationError` if `citation_number` is outside the
    supplied evidence's range -- the same failure mode as
    `validate_citations`, since both express the same "citation must
    refer to supplied evidence" invariant.
    """
    by_number = {item.citation_number: item for item in evidence}
    if citation_number not in by_number:
        raise CitationValidationError(
            f"Citation number {citation_number!r} does not refer to any of the "
            f"{len(evidence)} supplied evidence item(s)."
        )
    return by_number[citation_number]
