"""Phase 4 Step 2 evaluation metrics -- measurement only.

Five orthogonal metric families, each with its own immutable report and
no shared opaque "RAG score":

* **retrieval** (`evaluate_retrieval`) -- fully deterministic: required
  source Hit@k / Recall@k / Complete@k, reciprocal rank, identifier
  recall@k. Never reads a native retrieval score.
* **correctness** (`evaluate_correctness`) -- two orthogonal signals from
  one semantic judge: `expected_fact_score` (classify each golden
  `expected_fact`, then deterministic Python maps + means) and
  `has_golden_contradiction` (does the answer add a material claim that
  *conflicts* with the golden truth -- a claim merely absent from the
  non-exhaustive benchmark is not a contradiction). No numeric penalty is
  combined in. Golden facts are authoritative; retrieved evidence is
  never shown.
* **faithfulness** (`evaluate_faithfulness`) -- semantic judge extracts +
  classifies each material answer claim against the evidence supplied to
  generation; deterministic Python maps + means. The golden answer is
  never consulted.
* **citation accuracy** (`evaluate_citation_accuracy`) -- fully
  deterministic, from the production `CitationVerificationReport`:
  semantic support score, fully-supported rate, golden-source match rate,
  required-source citation recall.
* **abstention** (`evaluate_abstention` + `aggregate_abstention`) -- fully
  deterministic: policy answer/abstain decision vs golden answerability.

Nothing here runs retrieval, generation, judging, confidence, or the
abstention policy. Metrics OBSERVE `GoldenQACase` / retrieval results /
`FinalAnswer` -- they never modify production behaviour or the dataset.
A genuinely non-applicable metric is `None`, never `0.0`.
"""

from .abstention import aggregate_abstention, evaluate_abstention
from .citations import evaluate_citation_accuracy
from .correctness import (
    CorrectnessJudge,
    RawCorrectnessAssessment,
    RawFactVerdict,
    RawGoldenContradiction,
    evaluate_correctness,
)
from .faithfulness import FaithfulnessJudge, RawClaimVerdict, evaluate_faithfulness
from .models import (
    CLAIM_VERDICT_SCORES,
    FACT_VERDICT_SCORES,
    AbstentionAggregate,
    AbstentionMetrics,
    CitationMetrics,
    ClaimAssessment,
    ClaimVerdict,
    CorrectnessReport,
    FactAssessment,
    FactVerdict,
    FaithfulnessReport,
    GoldenContradiction,
    RetrievalMetrics,
)
from .openai_judge import OpenAIEvaluationJudge
from .prompts import (
    CORRECTNESS_SYSTEM_PROMPT,
    FAITHFULNESS_SYSTEM_PROMPT,
    build_correctness_user_prompt,
    build_faithfulness_user_prompt,
)
from .retrieval import RetrievedChunk, evaluate_retrieval

__all__ = [
    "CLAIM_VERDICT_SCORES",
    "CORRECTNESS_SYSTEM_PROMPT",
    "FACT_VERDICT_SCORES",
    "FAITHFULNESS_SYSTEM_PROMPT",
    "AbstentionAggregate",
    "AbstentionMetrics",
    "CitationMetrics",
    "ClaimAssessment",
    "ClaimVerdict",
    "CorrectnessJudge",
    "CorrectnessReport",
    "FactAssessment",
    "FactVerdict",
    "FaithfulnessJudge",
    "FaithfulnessReport",
    "GoldenContradiction",
    "OpenAIEvaluationJudge",
    "RawClaimVerdict",
    "RawCorrectnessAssessment",
    "RawFactVerdict",
    "RawGoldenContradiction",
    "RetrievalMetrics",
    "RetrievedChunk",
    "aggregate_abstention",
    "build_correctness_user_prompt",
    "build_faithfulness_user_prompt",
    "evaluate_abstention",
    "evaluate_citation_accuracy",
    "evaluate_correctness",
    "evaluate_faithfulness",
    "evaluate_retrieval",
]
