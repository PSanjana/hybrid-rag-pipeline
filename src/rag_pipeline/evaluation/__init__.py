"""Offline evaluation support for the RAG pipeline (Phase 4).

**Step 1** provides the *golden Q&A dataset*: a hand-authored,
version-controlled set of questions over the committed `data/sample/`
corpus, each with manually-grounded expected truth (atomic facts, source
documents, technical identifiers) and an answerable/unanswerable label.

**Step 2** adds *evaluation metrics* -- measurement only, no benchmark
run. Five orthogonal families, each with its own immutable report and no
single opaque score:

* `evaluate_retrieval` -- deterministic required-source Hit@k / Recall@k /
  Complete@k, reciprocal rank, identifier recall@k (never a native score).
* `evaluate_correctness` -- a semantic judge classifies each golden
  `expected_fact` (-> `expected_fact_score`) and separately flags answer
  claims that directly conflict with the golden truth
  (-> `has_golden_contradiction`); deterministic Python maps + means, no
  numeric penalty for contradictions. Retrieved evidence is never shown
  to it, and a claim merely absent from the golden facts is not a
  contradiction.
* `evaluate_faithfulness` -- a semantic judge classifies each material
  answer claim against the evidence supplied to generation; the golden
  answer is never consulted.
* `evaluate_citation_accuracy` -- deterministic, from the production
  `CitationVerificationReport`.
* `evaluate_abstention` / `aggregate_abstention` -- deterministic policy
  answer/abstain decision vs golden answerability.

Nothing in this package runs the RAG pipeline, tunes a weight/threshold,
or modifies the dataset. A non-applicable metric is `None`, never `0.0`.
"""

from .dataset import (
    DatasetValidationReport,
    default_golden_dataset_path,
    default_sample_corpus_dir,
    load_and_validate_golden_dataset,
    load_golden_dataset,
    parse_golden_case,
    validate_dataset,
)
from .exceptions import (
    EvaluationError,
    EvaluationJudgeError,
    EvaluationJudgeOutputError,
    GoldenDatasetError,
    MetricInputError,
)
from .metrics import (
    AbstentionAggregate,
    AbstentionMetrics,
    CitationMetrics,
    ClaimAssessment,
    ClaimVerdict,
    CorrectnessJudge,
    CorrectnessReport,
    FactAssessment,
    FactVerdict,
    FaithfulnessJudge,
    FaithfulnessReport,
    GoldenContradiction,
    OpenAIEvaluationJudge,
    RawClaimVerdict,
    RawCorrectnessAssessment,
    RawFactVerdict,
    RawGoldenContradiction,
    RetrievalMetrics,
    RetrievedChunk,
    aggregate_abstention,
    evaluate_abstention,
    evaluate_citation_accuracy,
    evaluate_correctness,
    evaluate_faithfulness,
    evaluate_retrieval,
)
from .models import Answerability, Difficulty, GoldenQACase, QuestionType

__all__ = [
    "AbstentionAggregate",
    "AbstentionMetrics",
    "Answerability",
    "CitationMetrics",
    "ClaimAssessment",
    "ClaimVerdict",
    "CorrectnessJudge",
    "CorrectnessReport",
    "DatasetValidationReport",
    "Difficulty",
    "EvaluationError",
    "EvaluationJudgeError",
    "EvaluationJudgeOutputError",
    "FactAssessment",
    "FactVerdict",
    "FaithfulnessJudge",
    "FaithfulnessReport",
    "GoldenContradiction",
    "GoldenDatasetError",
    "GoldenQACase",
    "MetricInputError",
    "OpenAIEvaluationJudge",
    "QuestionType",
    "RawClaimVerdict",
    "RawCorrectnessAssessment",
    "RawFactVerdict",
    "RawGoldenContradiction",
    "RetrievalMetrics",
    "RetrievedChunk",
    "aggregate_abstention",
    "default_golden_dataset_path",
    "default_sample_corpus_dir",
    "evaluate_abstention",
    "evaluate_citation_accuracy",
    "evaluate_correctness",
    "evaluate_faithfulness",
    "evaluate_retrieval",
    "load_and_validate_golden_dataset",
    "load_golden_dataset",
    "parse_golden_case",
    "validate_dataset",
]
