"""Domain-specific exceptions for the evaluation layer (Phase 4)."""


class EvaluationError(Exception):
    """Base class for all evaluation-layer errors."""


class GoldenDatasetError(EvaluationError):
    """Raised when a golden-evaluation record or the dataset as a whole is malformed.

    Covers a single unparseable/invalid JSONL record, a structural
    problem across records (duplicate case IDs), a violated answerability
    or multi-document invariant, and dataset-level or corpus-grounding
    checks. The golden dataset is a hand-authored source of truth, so
    every problem is surfaced loudly rather than skipped or repaired.
    """


class MetricInputError(EvaluationError):
    """Raised when a deterministic evaluation metric is handed inconsistent/malformed input.

    Every metric in Phase 4 Step 2 is a trust boundary: it never assumes
    the `GoldenQACase` / `FinalAnswer` / retrieval-result set it receives
    were produced together, and fails clearly rather than emit a
    misleading number from contradictory data (e.g. an `ANSWERED`
    `FinalAnswer` whose verification report has zero citations, a `k < 1`,
    or a cited number that resolves to no evidence).
    """


class EvaluationJudgeError(EvaluationError):
    """Raised when a semantic evaluation-judge provider is misconfigured, fails, or returns nothing.

    Mirrors `generation.CitationJudgeError`: a missing `OPENAI_API_KEY`
    fails fast when the production judge is instantiated, and every
    provider/transport failure is wrapped with its cause preserved rather
    than surfaced raw.
    """


class EvaluationJudgeOutputError(EvaluationError):
    """Raised when a semantic evaluation judge's structured output does not match what was asked.

    The correctness judge must return exactly one verdict per numbered
    golden fact (IDs `1..N`, no gaps/dupes/extras); the faithfulness
    judge must return a contiguous `1..M` list of at least one material
    claim. A malformed result -- wrong ID set, non-enum verdict, blank
    rationale, bool ID, zero claims -- raises this, never a raw
    `KeyError`/`TypeError`.
    """
