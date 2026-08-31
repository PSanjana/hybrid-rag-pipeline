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
