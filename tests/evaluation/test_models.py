"""Schema / answerability-invariant tests for parse_golden_case (offline)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import pytest

from rag_pipeline.evaluation import (
    Answerability,
    Difficulty,
    GoldenDatasetError,
    GoldenQACase,
    QuestionType,
    parse_golden_case,
)

Record = Callable[..., dict[str, object]]


# --- model / schema -----------------------------------------------------------------


def test_valid_answerable_case_accepted(answerable_record: Record) -> None:
    case = parse_golden_case(answerable_record())
    assert isinstance(case, GoldenQACase)
    assert case.answerability is Answerability.ANSWERABLE
    assert case.question_type is QuestionType.DIRECT_FACTUAL
    assert case.difficulty is Difficulty.EASY
    assert case.expected_source_files == ("database-operations.md",)


def test_valid_exact_identifier_case_accepted(exact_identifier_record: Record) -> None:
    case = parse_golden_case(exact_identifier_record())
    assert case.question_type is QuestionType.EXACT_IDENTIFIER
    assert case.expected_identifiers == ("ERR_DB_1042",)


def test_valid_unanswerable_case_accepted(unanswerable_record: Record) -> None:
    case = parse_golden_case(unanswerable_record())
    assert case.answerability is Answerability.UNANSWERABLE
    assert case.question_type is QuestionType.UNANSWERABLE_ABSENT
    assert case.expected_answer is None
    assert case.expected_facts == ()
    assert case.expected_source_files == ()


def test_case_is_immutable(answerable_record: Record) -> None:
    case = parse_golden_case(answerable_record())
    with pytest.raises(dataclasses.FrozenInstanceError):
        case.question = "mutated"  # type: ignore[misc]


def test_empty_id_rejected(answerable_record: Record) -> None:
    with pytest.raises(GoldenDatasetError, match="'id'"):
        parse_golden_case(answerable_record(id="  "))


def test_empty_question_rejected(answerable_record: Record) -> None:
    with pytest.raises(GoldenDatasetError, match="'question'"):
        parse_golden_case(answerable_record(question=""))


def test_malformed_answerability_rejected(answerable_record: Record) -> None:
    with pytest.raises(GoldenDatasetError, match="answerability"):
        parse_golden_case(answerable_record(answerability="maybe"))


def test_malformed_category_rejected(answerable_record: Record) -> None:
    with pytest.raises(GoldenDatasetError, match="question_type"):
        parse_golden_case(answerable_record(question_type="lexical"))


def test_malformed_difficulty_rejected(answerable_record: Record) -> None:
    with pytest.raises(GoldenDatasetError, match="difficulty"):
        parse_golden_case(answerable_record(difficulty="trivial"))


def test_unknown_field_rejected(answerable_record: Record) -> None:
    with pytest.raises(GoldenDatasetError, match="unknown field"):
        parse_golden_case(answerable_record(expected_sources=["x.md"]))


def test_missing_required_field_rejected(answerable_record: Record) -> None:
    record = answerable_record()
    del record["difficulty"]
    with pytest.raises(GoldenDatasetError, match="missing required field"):
        parse_golden_case(record)


def test_non_bool_multi_document_flag_rejected(answerable_record: Record) -> None:
    with pytest.raises(GoldenDatasetError, match="requires_multi_document_reasoning"):
        parse_golden_case(answerable_record(requires_multi_document_reasoning="yes"))


def test_duplicate_entries_in_string_tuple_rejected(answerable_record: Record) -> None:
    with pytest.raises(GoldenDatasetError, match="duplicate"):
        parse_golden_case(
            answerable_record(expected_source_files=["api-error-codes.txt", "api-error-codes.txt"])
        )


def test_acceptable_source_overlapping_expected_source_rejected(answerable_record: Record) -> None:
    with pytest.raises(GoldenDatasetError, match="both expected_source_files"):
        parse_golden_case(answerable_record(acceptable_source_files=["database-operations.md"]))


# --- answerability invariants ------------------------------------------------------


def test_answerable_case_requires_expected_answer(answerable_record: Record) -> None:
    with pytest.raises(GoldenDatasetError, match="requires an expected_answer"):
        parse_golden_case(answerable_record(expected_answer=None))


def test_answerable_case_requires_expected_facts(answerable_record: Record) -> None:
    with pytest.raises(GoldenDatasetError, match="requires at least one expected_facts"):
        parse_golden_case(answerable_record(expected_facts=[]))


def test_answerable_case_requires_at_least_one_source(answerable_record: Record) -> None:
    with pytest.raises(GoldenDatasetError, match="requires at least one expected_source_files"):
        parse_golden_case(answerable_record(expected_source_files=[]))


def test_answerable_case_cannot_use_unanswerable_absent_type(answerable_record: Record) -> None:
    with pytest.raises(GoldenDatasetError, match="must not use question_type"):
        parse_golden_case(answerable_record(question_type="unanswerable_absent"))


def test_unanswerable_case_rejects_expected_answer(unanswerable_record: Record) -> None:
    with pytest.raises(GoldenDatasetError, match="must not carry an authoritative expected_answer"):
        parse_golden_case(unanswerable_record(expected_answer="It has 4 spaces."))


def test_unanswerable_case_rejects_expected_facts(unanswerable_record: Record) -> None:
    with pytest.raises(GoldenDatasetError, match="must not carry expected_facts"):
        parse_golden_case(unanswerable_record(expected_facts=["there are 4 spaces"]))


def test_unanswerable_case_rejects_expected_source_files(unanswerable_record: Record) -> None:
    with pytest.raises(GoldenDatasetError, match="must not claim expected/acceptable source"):
        parse_golden_case(unanswerable_record(expected_source_files=["employee-handbook.pdf"]))


def test_unanswerable_case_must_use_absent_type(unanswerable_record: Record) -> None:
    with pytest.raises(GoldenDatasetError, match="must use question_type"):
        parse_golden_case(unanswerable_record(question_type="direct_factual"))


def test_unanswerable_case_must_not_require_multi_document(unanswerable_record: Record) -> None:
    with pytest.raises(GoldenDatasetError, match="must not require multi-document"):
        parse_golden_case(unanswerable_record(requires_multi_document_reasoning=True))


# --- multi-document consistency --------------------------------------------------


def test_multi_document_flag_requires_two_sources(answerable_record: Record) -> None:
    with pytest.raises(GoldenDatasetError, match="fewer than two expected_source_files"):
        parse_golden_case(
            answerable_record(
                question_type="multi_document_reasoning",
                requires_multi_document_reasoning=True,
                expected_source_files=["api-error-codes.txt"],
            )
        )


def test_multi_document_type_requires_the_flag(answerable_record: Record) -> None:
    with pytest.raises(GoldenDatasetError, match="requires requires_multi_document_reasoning"):
        parse_golden_case(
            answerable_record(
                question_type="multi_document_reasoning",
                requires_multi_document_reasoning=False,
                expected_source_files=["api-error-codes.txt", "database-operations.md"],
            )
        )


def test_multi_document_case_with_two_sources_accepted(answerable_record: Record) -> None:
    case = parse_golden_case(
        answerable_record(
            question_type="multi_document_reasoning",
            requires_multi_document_reasoning=True,
            expected_source_files=["api-error-codes.txt", "database-operations.md"],
        )
    )
    assert case.requires_multi_document_reasoning is True
    assert len(case.expected_source_files) == 2


# --- source-label semantics: non-multi answerable => exactly one required source ---


def test_non_multi_answerable_with_two_expected_sources_rejected(answerable_record: Record) -> None:
    with pytest.raises(GoldenDatasetError, match="must list exactly one expected_source_files"):
        parse_golden_case(
            answerable_record(
                requires_multi_document_reasoning=False,
                expected_source_files=["database-operations.md", "backup-recovery.html"],
            )
        )


def test_non_multi_answerable_corroborating_source_belongs_in_acceptable(
    answerable_record: Record,
) -> None:
    # the same two documents, but one moved to acceptable_source_files -> valid
    case = parse_golden_case(
        answerable_record(
            requires_multi_document_reasoning=False,
            expected_source_files=["database-operations.md"],
            acceptable_source_files=["backup-recovery.html"],
        )
    )
    assert case.expected_source_files == ("database-operations.md",)
    assert case.acceptable_source_files == ("backup-recovery.html",)


# --- exact_identifier category semantics ------------------------------------------


def test_exact_identifier_case_requires_at_least_one_identifier(
    exact_identifier_record: Record,
) -> None:
    with pytest.raises(GoldenDatasetError, match="must list at least one expected_identifiers"):
        parse_golden_case(exact_identifier_record(expected_identifiers=[]))


def test_exact_identifier_case_requires_an_identifier_in_the_question(
    exact_identifier_record: Record,
) -> None:
    with pytest.raises(GoldenDatasetError, match="verbatim in the question"):
        parse_golden_case(
            exact_identifier_record(
                question="What is the database connection pool wait time?",
                expected_identifiers=["DATABASE_POOL_TIMEOUT"],
            )
        )


def test_exact_identifier_case_accepts_identifier_present_in_question_case_insensitively(
    exact_identifier_record: Record,
) -> None:
    case = parse_golden_case(
        exact_identifier_record(
            question="what does err_db_1042 actually mean?",
            expected_identifiers=["ERR_DB_1042"],
        )
    )
    assert case.question_type is QuestionType.EXACT_IDENTIFIER
