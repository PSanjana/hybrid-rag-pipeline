"""Shared fixtures/helpers for evaluation tests (offline, no pipeline, no network)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

Record = Callable[..., dict[str, object]]


def _answerable_record(**overrides: object) -> dict[str, object]:
    """A minimal valid ANSWERABLE case: single-source, non-multi, unconstrained category."""
    record: dict[str, object] = {
        "id": "case-1",
        "question": "How often are full PostgreSQL backups taken?",
        "answerability": "answerable",
        "question_type": "direct_factual",
        "difficulty": "easy",
        "requires_multi_document_reasoning": False,
        "expected_answer": "Full PostgreSQL backups run daily.",
        "expected_facts": ["Full PostgreSQL backups run daily"],
        "expected_source_files": ["database-operations.md"],
        "expected_identifiers": [],
    }
    record.update(overrides)
    return record


def _exact_identifier_record(**overrides: object) -> dict[str, object]:
    """A minimal valid ANSWERABLE `exact_identifier` case (identifier present in the question)."""
    record: dict[str, object] = {
        "id": "exact-1",
        "question": "What does ERR_DB_1042 mean?",
        "answerability": "answerable",
        "question_type": "exact_identifier",
        "difficulty": "easy",
        "requires_multi_document_reasoning": False,
        "expected_answer": "A database connection could not be obtained from the pool.",
        "expected_facts": ["ERR_DB_1042 means a database connection could not be obtained"],
        "expected_source_files": ["api-error-codes.txt"],
        "expected_identifiers": ["ERR_DB_1042"],
    }
    record.update(overrides)
    return record


def _unanswerable_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "id": "absent-1",
        "question": "How many visitor parking spaces does the office have?",
        "answerability": "unanswerable",
        "question_type": "unanswerable_absent",
        "difficulty": "easy",
        "requires_multi_document_reasoning": False,
        "expected_answer": None,
        "notes": "The corpus never mentions an office or parking.",
    }
    record.update(overrides)
    return record


@pytest.fixture
def answerable_record() -> Record:
    return _answerable_record


@pytest.fixture
def exact_identifier_record() -> Record:
    return _exact_identifier_record


@pytest.fixture
def unanswerable_record() -> Record:
    return _unanswerable_record


@pytest.fixture
def write_jsonl(tmp_path: Path) -> Callable[[list[dict[str, object]]], Path]:
    def _write(records: list[dict[str, object]]) -> Path:
        path = tmp_path / "golden_qa.jsonl"
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
        )
        return path

    return _write
