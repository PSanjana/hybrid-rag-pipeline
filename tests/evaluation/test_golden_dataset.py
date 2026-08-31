"""Assertions about the committed real golden dataset (data/eval/golden_qa.jsonl), offline."""

from __future__ import annotations

import re

import pytest

from rag_pipeline.evaluation import (
    Answerability,
    QuestionType,
    default_golden_dataset_path,
    default_sample_corpus_dir,
    load_and_validate_golden_dataset,
    load_golden_dataset,
    validate_dataset,
)

_SHA_RE = re.compile(r"\b[0-9a-f]{64}\b")


@pytest.fixture(scope="module")
def golden() -> tuple:
    return load_golden_dataset(default_golden_dataset_path())


def test_committed_dataset_passes_full_validation() -> None:
    # No network, no pipeline: pure file + corpus reads.
    cases = load_and_validate_golden_dataset()
    assert len(cases) >= 50


def test_dataset_has_at_least_50_cases(golden: tuple) -> None:
    assert len(golden) >= 50


def test_dataset_has_at_least_10_unanswerable_cases(golden: tuple) -> None:
    unanswerable = [c for c in golden if c.answerability is Answerability.UNANSWERABLE]
    assert len(unanswerable) >= 10


def test_answerable_majority_is_not_overwhelming(golden: tuple) -> None:
    answerable = sum(1 for c in golden if c.answerability is Answerability.ANSWERABLE)
    # sanity: negatives are a meaningful fraction, not a token handful
    assert (len(golden) - answerable) / len(golden) >= 0.15


def test_every_required_category_is_represented(golden: tuple) -> None:
    present = {c.question_type for c in golden}
    assert present == set(QuestionType)


def test_exact_identifier_cases_present(golden: tuple) -> None:
    exact = [c for c in golden if c.question_type is QuestionType.EXACT_IDENTIFIER]
    assert len(exact) >= 8
    # each such case names at least one identifier
    assert all(c.expected_identifiers for c in exact)


def test_semantic_paraphrase_cases_present(golden: tuple) -> None:
    assert sum(1 for c in golden if c.question_type is QuestionType.SEMANTIC_PARAPHRASE) >= 8


def test_multi_document_cases_present_and_consistent(golden: tuple) -> None:
    multi = [c for c in golden if c.question_type is QuestionType.MULTI_DOCUMENT_REASONING]
    assert len(multi) >= 8
    for case in multi:
        assert case.requires_multi_document_reasoning is True
        assert len(case.expected_source_files) >= 2


def test_overlap_ambiguity_cases_present(golden: tuple) -> None:
    assert sum(1 for c in golden if c.question_type is QuestionType.OVERLAP_AMBIGUITY) >= 6


def test_all_case_ids_unique(golden: tuple) -> None:
    ids = [c.id for c in golden]
    assert len(ids) == len(set(ids))


def test_every_source_reference_resolves_to_a_real_corpus_document(golden: tuple) -> None:
    real = {
        p.name
        for p in default_sample_corpus_dir().rglob("*")
        if p.is_file() and p.suffix.lower() in {".md", ".txt", ".html", ".pdf"}
    }
    for case in golden:
        for name in (*case.expected_source_files, *case.acceptable_source_files):
            assert name in real, f"{case.id}: unknown source {name!r}"


def test_every_corpus_document_is_exercised_by_at_least_one_case(golden: tuple) -> None:
    real = {
        p.name
        for p in default_sample_corpus_dir().rglob("*")
        if p.is_file() and p.suffix.lower() in {".md", ".txt", ".html", ".pdf"}
    }
    cited = {name for case in golden for name in case.expected_source_files}
    assert real <= cited, f"documents never used as a golden source: {sorted(real - cited)}"


def test_no_case_contains_authoritative_chunk_ids(golden: tuple) -> None:
    for case in golden:
        blob = " ".join(
            [
                case.id,
                case.question,
                case.expected_answer or "",
                case.notes or "",
                *case.expected_facts,
                *case.expected_identifiers,
                *case.expected_source_files,
                *case.acceptable_source_files,
                *case.tags,
            ]
        )
        assert _SHA_RE.search(blob) is None, f"{case.id}: contains a chunk-id-shaped token"


def test_answerable_cases_carry_facts_answer_and_sources(golden: tuple) -> None:
    for case in golden:
        if case.answerability is Answerability.ANSWERABLE:
            assert case.expected_answer
            assert case.expected_facts
            assert case.expected_source_files


def test_unanswerable_cases_carry_no_authoritative_truth(golden: tuple) -> None:
    for case in golden:
        if case.answerability is Answerability.UNANSWERABLE:
            assert case.expected_answer is None
            assert case.expected_facts == ()
            assert case.expected_source_files == ()
            assert case.acceptable_source_files == ()
            assert case.requires_multi_document_reasoning is False


def test_validation_report_counts_match_manual_tally(golden: tuple) -> None:
    report = validate_dataset(golden)
    assert report.ok
    assert report.total == len(golden)
    assert report.answerable + report.unanswerable == report.total
    assert sum(report.by_question_type.values()) == report.total
    assert sum(report.by_difficulty.values()) == report.total
    assert report.multi_document + report.single_document == report.total
