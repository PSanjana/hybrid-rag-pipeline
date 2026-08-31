"""Offline evaluation support for the RAG pipeline (Phase 4).

Step 1 provides only the *golden Q&A dataset*: a hand-authored,
version-controlled set of questions over the committed `data/sample/`
corpus, each with manually-grounded expected truth (atomic facts,
source documents, technical identifiers) and an answerable/unanswerable
label. Retrieval, correctness, faithfulness, citation, and abstention
metrics -- and any comparison of chunking strategies -- are later steps
and are NOT implemented here.

Nothing in this package runs the RAG pipeline or reads its outputs.
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
from .exceptions import EvaluationError, GoldenDatasetError
from .models import Answerability, Difficulty, GoldenQACase, QuestionType

__all__ = [
    "Answerability",
    "DatasetValidationReport",
    "Difficulty",
    "EvaluationError",
    "GoldenDatasetError",
    "GoldenQACase",
    "QuestionType",
    "default_golden_dataset_path",
    "default_sample_corpus_dir",
    "load_and_validate_golden_dataset",
    "load_golden_dataset",
    "parse_golden_case",
    "validate_dataset",
]
