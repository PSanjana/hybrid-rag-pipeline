"""Immutable typed model for one golden-evaluation Q&A case (Phase 4 Step 1).

This module defines the *shape* of a golden case only. Every rule --
schema, answerability invariants, multi-document consistency, and
corpus-grounding checks -- lives in `evaluation.dataset`, keeping the
model a dumb immutable container consistent with the rest of the
codebase (`Evidence`, `ConfidenceAssessment`, `FinalAnswer`, ...).

Golden truth is expressed as **source documents, atomic facts, and
technical identifiers**, never as chunk IDs: Phase 4 later compares the
fixed / recursive / semantic chunking strategies, and chunk IDs are a
function of the strategy and its boundaries, so they would not be stable
across the very comparison the dataset exists to support.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Answerability(StrEnum):
    """Whether the committed sample corpus actually supports an answer."""

    ANSWERABLE = "answerable"
    """The corpus contains the facts needed to answer; the pipeline is
    expected to produce a grounded substantive answer."""

    UNANSWERABLE = "unanswerable"
    """The corpus genuinely does not contain the information; the
    pipeline is expected to abstain gracefully."""


class QuestionType(StrEnum):
    """The benchmark category a case belongs to (used for balanced coverage)."""

    EXACT_IDENTIFIER = "exact_identifier"
    """Contains a rare exact token (e.g. `ERR_DB_1042`, `AUTH_TOKEN_TTL`);
    exercises BM25 lexical strength."""

    SEMANTIC_PARAPHRASE = "semantic_paraphrase"
    """Worded so it does not copy documentation phrasing; exercises dense
    semantic retrieval."""

    DIRECT_FACTUAL = "direct_factual"
    """A straightforward single-fact policy/product/operations lookup."""

    MULTI_DOCUMENT_REASONING = "multi_document_reasoning"
    """Needs facts from at least two documents to answer completely."""

    OVERLAP_AMBIGUITY = "overlap_ambiguity"
    """Related information appears in several documents but only one (or a
    subset) is authoritative for the detail asked; exercises reranking
    and citation correctness."""

    UNANSWERABLE_ABSENT = "unanswerable_absent"
    """The information is genuinely absent from every sample document."""


class Difficulty(StrEnum):
    """How hard the case is expected to be, by a fixed definition.

    EASY: a single explicit fact stated in one document.
    MEDIUM: a paraphrase, ambiguous terminology, or connecting multiple
        nearby facts within a document / small document set.
    HARD: multi-document reasoning, overlapping/competing evidence, or a
        subtle abstention judgment.
    """

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass(frozen=True, slots=True)
class GoldenQACase:
    """One manually-grounded evaluation question with its expected truth.

    Immutable. Constructed only by `evaluation.dataset.parse_golden_case`,
    which enforces every invariant; nothing here validates its own
    fields.

    Answerable cases carry `expected_answer` (human-readable),
    `expected_facts` (atomic claims a correct answer must contain -- the
    primary basis for future correctness scoring, more robust than string
    comparison), and at least one `expected_source_files` entry (the
    documents that actually support those facts, by exact ingestion
    basename). Unanswerable cases carry none of those and exist to
    exercise the abstention policy.

    `expected_identifiers` lists exact technical tokens whose retrieval
    matters for the case. `acceptable_source_files` names documents that
    would also be a legitimate citation for the same detail without being
    the authoritative one (useful for overlap/ambiguity cases).
    `requires_multi_document_reasoning` is a hard label: when true, the
    complete answer needs >= 2 documents.
    """

    id: str
    question: str
    answerability: Answerability
    question_type: QuestionType
    difficulty: Difficulty
    requires_multi_document_reasoning: bool
    expected_answer: str | None
    expected_facts: tuple[str, ...]
    expected_source_files: tuple[str, ...]
    expected_identifiers: tuple[str, ...]
    acceptable_source_files: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    notes: str | None = None
