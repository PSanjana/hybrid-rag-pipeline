"""Immutable typed reports and verdict enums for Phase 4 Step 2 evaluation metrics.

Design rules held across every report here:

* **Diagnostic separation.** There is no single opaque "RAG score". Retrieval,
  correctness, faithfulness, citation accuracy, and abstention each have their
  own report type. A correct answer built on failed retrieval stays
  distinguishable from a correct answer built on successful retrieval.
* **Raw verdicts are always kept.** Every semantic report retains the ordered
  categorical verdicts (`FactVerdict` / `ClaimVerdict`) *and* the deterministic
  score derived from them -- never only a float.
* **N/A is `None`, never `0.0`.** A metric that genuinely cannot be computed
  (retrieval source recall for an unanswerable case, identifier recall when the
  case has no identifiers, correctness/citation metrics when the policy
  abstained) reports `None`. `0.0` means the metric was applicable and the
  system scored zero -- a different fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

# --- semantic verdict vocabularies ------------------------------------------------


class FactVerdict(StrEnum):
    """One golden expected fact's coverage verdict from the correctness judge."""

    CORRECT = "correct"
    """The answer states this fact and it matches the golden fact."""

    PARTIALLY_CORRECT = "partially_correct"
    """The answer captures part of the fact but omits/softens a material detail."""

    MISSING = "missing"
    """The answer does not address this fact at all."""

    CONTRADICTED = "contradicted"
    """The answer states something that conflicts with this fact."""


FACT_VERDICT_SCORES: dict[FactVerdict, float] = {
    FactVerdict.CORRECT: 1.0,
    FactVerdict.PARTIALLY_CORRECT: 0.5,
    FactVerdict.MISSING: 0.0,
    FactVerdict.CONTRADICTED: 0.0,
}
"""Fixed initial deterministic mapping. The judge classifies; Python maps + means."""


class ClaimVerdict(StrEnum):
    """One material answer claim's support verdict from the faithfulness judge.

    Deliberately a separate enum from `generation.CitationVerdict`, even
    though the four names coincide: faithfulness judges *answer claims*
    against *supplied evidence*, not citation occurrences, and the two
    vocabularies are free to diverge later.
    """

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


CLAIM_VERDICT_SCORES: dict[ClaimVerdict, float] = {
    ClaimVerdict.SUPPORTED: 1.0,
    ClaimVerdict.PARTIALLY_SUPPORTED: 0.5,
    ClaimVerdict.UNSUPPORTED: 0.0,
    ClaimVerdict.CONTRADICTED: 0.0,
}
"""Fixed initial deterministic mapping. The judge classifies; Python maps + means."""


# --- retrieval -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Deterministic retrieval-relevance signals for one golden case at a fixed `k`.

    All four source signals are `None` for an unanswerable case (there is
    no expected-source truth). `identifier_recall_at_k` is `None` when the
    case lists no `expected_identifiers`. `reciprocal_rank` ranges over
    the *entire* supplied result sequence (standard reciprocal rank),
    independent of `k`; every other number is scoped to the top `k`.

    `required_source_hit_at_k`, `required_source_recall_at_k`, and
    `complete_required_source_retrieval_at_k` are kept as three distinct
    fields on purpose: for a multi-document case, Hit@k can be 1.0 while
    Complete@k is `False` (only one of two required documents was found).
    """

    k: int
    required_source_hit_at_k: float | None
    required_source_recall_at_k: float | None
    complete_required_source_retrieval_at_k: bool | None
    reciprocal_rank: float | None
    identifier_recall_at_k: float | None
    required_sources_found: tuple[str, ...] = ()
    required_sources_missing: tuple[str, ...] = ()
    identifiers_found: tuple[str, ...] = ()
    identifiers_missing: tuple[str, ...] = ()


# --- correctness --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FactAssessment:
    """One golden expected fact, its judge verdict, and the judge's rationale."""

    fact_id: int
    fact_text: str
    verdict: FactVerdict
    rationale: str


@dataclass(frozen=True, slots=True)
class GoldenContradiction:
    """One material answer claim that DIRECTLY CONFLICTS with the supplied golden truth.

    This is NOT "a claim absent from the golden facts" -- the golden facts
    are not exhaustive, so an extra statement is only a
    `GoldenContradiction` when it conflicts with something that can
    actually be established from the golden `expected_facts` /
    `expected_answer`. `conflicting_fact_ids` optionally records which
    numbered expected facts it clashes with (may be empty when the
    conflict is against the reference `expected_answer` as a whole).
    """

    contradiction_id: int
    claim_text: str
    rationale: str
    conflicting_fact_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class CorrectnessReport:
    """How well the substantive answer covers golden `expected_facts`, plus contradiction flags.

    `applicable` is `True` only for an ANSWERABLE golden case whose final
    policy decision was `ANSWERED`. Otherwise both score fields are `None`,
    the tuples are empty, and `not_applicable_reason` says why (a false
    abstention on an answerable case is measured by the abstention metric,
    not counted as correctness 0.0).

    `expected_fact_score` is the authoritative numeric coverage score:
    ``mean(FACT_VERDICT_SCORES[verdict])`` over every numbered expected
    fact (`CORRECT`=1.0, `PARTIALLY_CORRECT`=0.5, `MISSING`=0.0,
    `CONTRADICTED`=0.0). `score` holds the *same value* and is kept only
    for backwards clarity -- **the float alone is not a complete
    correctness decision**. A complete decision must also inspect
    `has_golden_contradiction`: an answer can state every expected fact
    correctly (`expected_fact_score == 1.0`) yet still contain an extra
    material claim that contradicts the golden truth. No numeric penalty
    for contradictions is applied here -- how to combine the coverage
    score with the contradiction signal is deferred to later Phase 4
    analysis.
    """

    applicable: bool
    score: float | None
    expected_fact_score: float | None
    fact_assessments: tuple[FactAssessment, ...] = ()
    correct_count: int = 0
    partially_correct_count: int = 0
    missing_count: int = 0
    contradicted_count: int = 0
    golden_contradictions: tuple[GoldenContradiction, ...] = ()
    golden_contradiction_count: int = 0
    has_golden_contradiction: bool = False
    not_applicable_reason: str | None = None


# --- faithfulness -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClaimAssessment:
    """One material claim extracted from the answer, its verdict, and rationale."""

    claim_id: int
    claim_text: str
    verdict: ClaimVerdict
    rationale: str


@dataclass(frozen=True, slots=True)
class FaithfulnessReport:
    """Whether the answer's material claims are supported by the evidence given to generation.

    `applicable` is `True` only when the final policy decision was
    `ANSWERED` (the rejected draft behind an abstention is never scored as
    though it had been shown). It is independent of golden answerability:
    an erroneously-answered *unanswerable* case can still be scored for
    faithfulness against its supplied evidence. This is NOT correctness --
    the golden expected answer is never consulted here.
    """

    applicable: bool
    score: float | None
    claim_assessments: tuple[ClaimAssessment, ...] = ()
    supported_count: int = 0
    partially_supported_count: int = 0
    unsupported_count: int = 0
    contradicted_count: int = 0
    not_applicable_reason: str | None = None


# --- citation accuracy ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CitationMetrics:
    """Deterministic citation-accuracy signals derived from the production verification report.

    `applicable` is `True` only when the final decision was `ANSWERED`.
    `semantic_citation_support_score` is the mean mapped verdict over
    citation *occurrences* and is independent of the Step 3 confidence
    composite (the confidence score is never read here).
    `cited_source_golden_match_rate` is named as a GOLDEN-SOURCE match,
    not universal precision: the benchmark does not assert that every
    document it does not list is irrelevant.
    """

    applicable: bool
    semantic_citation_support_score: float | None
    fully_supported_citation_rate: float | None
    cited_source_golden_match_rate: float | None
    required_source_citation_recall: float | None
    total_citation_occurrences: int = 0
    supported_count: int = 0
    partially_supported_count: int = 0
    unsupported_count: int = 0
    contradicted_count: int = 0
    unique_cited_source_files: tuple[str, ...] = ()
    matched_cited_source_files: tuple[str, ...] = ()
    unmatched_cited_source_files: tuple[str, ...] = ()
    required_sources_cited: tuple[str, ...] = ()
    required_sources_not_cited: tuple[str, ...] = ()
    not_applicable_reason: str | None = None


# --- abstention ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AbstentionMetrics:
    """Whether the deterministic policy's answer/abstain decision matched golden expectation.

    Derived purely from `FinalAnswer.abstained` vs the golden
    `Answerability` -- never inferred from the confidence score. A
    `false_abstention` is an answerable case the policy declined; a
    `false_answer` is an unanswerable case the policy answered.
    """

    expected_abstain: bool
    actual_abstain: bool
    decision_correct: bool
    false_abstention: bool
    false_answer: bool


@dataclass(frozen=True, slots=True)
class AbstentionAggregate:
    """Pure aggregation over many `AbstentionMetrics`. Zero-denominator rates are `None`."""

    total: int
    total_answerable: int
    total_unanswerable: int
    decision_accuracy: float | None
    answerable_coverage: float | None
    false_abstention_rate: float | None
    unanswerable_abstention_recall: float | None
    false_answer_rate: float | None
