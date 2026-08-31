"""Loader and dataset-level / corpus validation tests (offline, synthetic corpus)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from rag_pipeline.evaluation import (
    GoldenDatasetError,
    load_golden_dataset,
    parse_golden_case,
    validate_dataset,
)
from rag_pipeline.evaluation.dataset import _CHUNK_ID_RE

Record = Callable[..., dict[str, object]]
WriteJsonl = Callable[[list[dict[str, object]]], Path]


@pytest.fixture
def fake_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "sample"
    (root / "engineering").mkdir(parents=True)
    (root / "product").mkdir(parents=True)
    (root / "engineering" / "api-error-codes.txt").write_text(
        "Code: ERR_DB_1042 - no DB connection before DATABASE_POOL_TIMEOUT.\n"
        "Code: ERR_AUTH_4017 - AUTH_TOKEN_TTL elapsed.\n",
        encoding="utf-8",
    )
    (root / "engineering" / "database-operations.md").write_text(
        "# Database\nDATABASE_POOL_SIZE default 20. ERR_DB_1042 is pool exhaustion.\n",
        encoding="utf-8",
    )
    (root / "product" / "authentication-api.md").write_text(
        "Access tokens expire after 60 minutes. AUTH_TOKEN_TTL is 3600 in production.\n",
        encoding="utf-8",
    )
    return root


# --- loader ------------------------------------------------------------------------


def test_jsonl_loads_and_preserves_order(
    write_jsonl: WriteJsonl, answerable_record: Record, unanswerable_record: Record
) -> None:
    path = write_jsonl(
        [
            answerable_record(id="b-second"),
            unanswerable_record(id="a-first"),
            answerable_record(id="c-third", question="Another?"),
        ]
    )
    cases = load_golden_dataset(path)
    assert [c.id for c in cases] == ["b-second", "a-first", "c-third"]


def test_blank_lines_are_ignored(
    write_jsonl: WriteJsonl, answerable_record: Record, tmp_path: Path
) -> None:
    path = write_jsonl([answerable_record()])
    path.write_text(path.read_text(encoding="utf-8") + "\n   \n", encoding="utf-8")
    assert len(load_golden_dataset(path)) == 1


def test_duplicate_ids_rejected(write_jsonl: WriteJsonl, answerable_record: Record) -> None:
    path = write_jsonl([answerable_record(id="dup"), answerable_record(id="dup", question="Q2?")])
    with pytest.raises(GoldenDatasetError, match="duplicate case id"):
        load_golden_dataset(path)


def test_malformed_json_rejected(tmp_path: Path) -> None:
    path = tmp_path / "golden_qa.jsonl"
    path.write_text('{"id": "x", not json}\n', encoding="utf-8")
    with pytest.raises(GoldenDatasetError, match="not valid JSON"):
        load_golden_dataset(path)


def test_missing_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(GoldenDatasetError, match="not found"):
        load_golden_dataset(tmp_path / "does-not-exist.jsonl")


def test_non_object_line_rejected(tmp_path: Path) -> None:
    path = tmp_path / "golden_qa.jsonl"
    path.write_text('["not", "an", "object"]\n', encoding="utf-8")
    with pytest.raises(GoldenDatasetError, match="must be a JSON object"):
        load_golden_dataset(path)


def test_load_is_deterministic(write_jsonl: WriteJsonl, answerable_record: Record) -> None:
    path = write_jsonl([answerable_record(id=f"c{i}", question=f"Q{i}?") for i in range(5)])
    assert [c.id for c in load_golden_dataset(path)] == [c.id for c in load_golden_dataset(path)]


# --- corpus validation -----------------------------------------------------------


def _valid_dataset(answerable_record: Record, unanswerable_record: Record) -> list:
    cases = []
    for i in range(40):
        cases.append(
            parse_golden_case(answerable_record(id=f"ans-{i}", question=f"Answerable {i}?"))
        )
    for i in range(12):
        cases.append(
            parse_golden_case(unanswerable_record(id=f"absent-{i}", question=f"Absent {i}?"))
        )
    # one of each remaining question_type so category coverage passes
    for qtype in ("semantic_paraphrase", "direct_factual", "overlap_ambiguity"):
        cases.append(
            parse_golden_case(
                answerable_record(id=f"type-{qtype}", question=f"{qtype}?", question_type=qtype)
            )
        )
    cases.append(
        parse_golden_case(
            answerable_record(
                id="type-exact",
                question="What does ERR_DB_1042 mean?",
                question_type="exact_identifier",
                expected_source_files=["api-error-codes.txt"],
                expected_identifiers=["ERR_DB_1042"],
            )
        )
    )
    cases.append(
        parse_golden_case(
            answerable_record(
                id="type-multi",
                question="multi?",
                question_type="multi_document_reasoning",
                requires_multi_document_reasoning=True,
                expected_source_files=["api-error-codes.txt", "database-operations.md"],
            )
        )
    )
    return cases


def test_expected_source_file_exists(
    fake_corpus: Path, answerable_record: Record, unanswerable_record: Record
) -> None:
    cases = _valid_dataset(answerable_record, unanswerable_record)
    report = validate_dataset(cases, sample_corpus_dir=fake_corpus)
    assert report.problems == ()
    assert report.ok


def test_nonexistent_source_rejected(fake_corpus: Path, answerable_record: Record) -> None:
    case = parse_golden_case(answerable_record(expected_source_files=["ghost-doc.md"]))
    report = validate_dataset(
        [case], sample_corpus_dir=fake_corpus, min_cases=1, min_unanswerable=0
    )
    assert any("ghost-doc.md" in p and "not a file" in p for p in report.problems)


def test_multi_document_case_with_one_source_rejected_by_dataset_validation(
    fake_corpus: Path, answerable_record: Record
) -> None:
    # bypass parse_golden_case's own check by hand-building the frozen model
    from rag_pipeline.evaluation.models import (
        Answerability,
        Difficulty,
        GoldenQACase,
        QuestionType,
    )

    case = GoldenQACase(
        id="bad-multi",
        question="q?",
        answerability=Answerability.ANSWERABLE,
        question_type=QuestionType.MULTI_DOCUMENT_REASONING,
        difficulty=Difficulty.HARD,
        requires_multi_document_reasoning=True,
        expected_answer="a",
        expected_facts=("f",),
        expected_source_files=("api-error-codes.txt",),
        expected_identifiers=(),
    )
    report = validate_dataset(
        [case], sample_corpus_dir=fake_corpus, min_cases=1, min_unanswerable=0
    )
    assert any(
        "requires_multi_document_reasoning but 1 expected_source_files" in p
        for p in report.problems
    )


def test_identifier_validation_flags_token_absent_from_source(
    fake_corpus: Path, exact_identifier_record: Record
) -> None:
    # identifier is in the question (parser passes) but not in the source corpus
    case = parse_golden_case(
        exact_identifier_record(
            id="id-missing",
            question="What does ERR_NOT_REAL_9999 mean?",
            expected_identifiers=["ERR_NOT_REAL_9999"],
        )
    )
    report = validate_dataset(
        [case], sample_corpus_dir=fake_corpus, min_cases=1, min_unanswerable=0
    )
    assert any("ERR_NOT_REAL_9999" in p and "does not occur" in p for p in report.problems)


def test_identifier_validation_passes_when_token_present_in_source(
    fake_corpus: Path, exact_identifier_record: Record
) -> None:
    case = parse_golden_case(
        exact_identifier_record(
            id="id-present",
            question="What is DATABASE_POOL_SIZE?",
            expected_source_files=["database-operations.md"],
            expected_identifiers=["DATABASE_POOL_SIZE"],
        )
    )
    report = validate_dataset(
        [case], sample_corpus_dir=fake_corpus, min_cases=1, min_unanswerable=0
    )
    assert not any("does not occur" in p for p in report.problems)
    assert not any("not a file" in p for p in report.problems)


def test_min_case_and_unanswerable_thresholds_enforced(
    fake_corpus: Path, answerable_record: Record
) -> None:
    report = validate_dataset(
        [parse_golden_case(answerable_record())], sample_corpus_dir=fake_corpus
    )
    assert any("at least 50 required" in p for p in report.problems)
    assert any("at least 10 required" in p for p in report.problems)


def test_missing_category_flagged(fake_corpus: Path, answerable_record: Record) -> None:
    report = validate_dataset(
        [parse_golden_case(answerable_record())],
        sample_corpus_dir=fake_corpus,
        min_cases=1,
        min_unanswerable=0,
    )
    assert any("no cases for question_type" in p for p in report.problems)


def test_chunk_id_shaped_token_rejected(fake_corpus: Path, answerable_record: Record) -> None:
    sha = "a" * 64
    case = parse_golden_case(answerable_record(notes=f"see chunk {sha}"))
    report = validate_dataset(
        [case], sample_corpus_dir=fake_corpus, min_cases=1, min_unanswerable=0
    )
    assert any("chunk-id-shaped token" in p for p in report.problems)


def test_raise_for_problems_raises_on_failure(fake_corpus: Path, answerable_record: Record) -> None:
    report = validate_dataset(
        [parse_golden_case(answerable_record())],
        sample_corpus_dir=fake_corpus,
        min_cases=1,
        min_unanswerable=0,
    )
    with pytest.raises(GoldenDatasetError, match="validation failed"):
        report.raise_for_problems()


def test_chunk_id_regex_matches_sha_but_not_short_hex() -> None:
    assert _CHUNK_ID_RE.search("x " + "0" * 64 + " y")
    assert _CHUNK_ID_RE.search("deadbeef" * 4) is None  # 32 hex chars, not 64


def test_non_string_scalar_field_rejected(answerable_record: Record) -> None:
    with pytest.raises(GoldenDatasetError, match="'id'.*must be a string"):
        parse_golden_case(answerable_record(id=123))


def test_non_list_collection_field_rejected(answerable_record: Record) -> None:
    with pytest.raises(GoldenDatasetError, match="'expected_facts'.*must be a list"):
        parse_golden_case(answerable_record(expected_facts="a single string"))


def test_validate_dataset_flags_duplicate_ids_on_a_hand_built_list(
    fake_corpus: Path, answerable_record: Record
) -> None:
    case = parse_golden_case(answerable_record(id="dup"))
    report = validate_dataset(
        [case, case], sample_corpus_dir=fake_corpus, min_cases=1, min_unanswerable=0
    )
    assert any("duplicate case id(s): dup" in p for p in report.problems)


def test_missing_sample_corpus_directory_is_reported(
    tmp_path: Path, answerable_record: Record
) -> None:
    report = validate_dataset(
        [parse_golden_case(answerable_record())],
        sample_corpus_dir=tmp_path / "no-such-dir",
        min_cases=1,
        min_unanswerable=0,
    )
    assert any("sample corpus directory not found" in p for p in report.problems)


def test_identifier_check_cannot_verify_pdf_source(
    fake_corpus: Path, exact_identifier_record: Record
) -> None:
    (fake_corpus / "people").mkdir()
    (fake_corpus / "people" / "handbook.pdf").write_bytes(b"%PDF-1.4 fake")
    case = parse_golden_case(
        exact_identifier_record(
            id="pdf-id",
            expected_source_files=["handbook.pdf"],
            expected_identifiers=["ERR_DB_1042"],
        )
    )
    report = validate_dataset(
        [case], sample_corpus_dir=fake_corpus, min_cases=1, min_unanswerable=0
    )
    # PDF text cannot be read for a substring check, so the identifier is
    # treated as not found rather than silently passing.
    assert any("ERR_DB_1042" in p and "does not occur" in p for p in report.problems)


# --- ambiguous duplicate corpus basenames --------------------------------------------


def test_duplicate_corpus_basename_is_reported(tmp_path: Path, answerable_record: Record) -> None:
    root = tmp_path / "sample"
    (root / "domain_a").mkdir(parents=True)
    (root / "domain_b").mkdir(parents=True)
    (root / "domain_a" / "runbook.md").write_text("A runbook.\n", encoding="utf-8")
    (root / "domain_b" / "runbook.md").write_text("Another runbook.\n", encoding="utf-8")
    (root / "domain_a" / "database-operations.md").write_text(
        "Backups run daily.\n", encoding="utf-8"
    )

    case = parse_golden_case(answerable_record())
    report = validate_dataset([case], sample_corpus_dir=root, min_cases=1, min_unanswerable=0)
    problem = next((p for p in report.problems if "ambiguous corpus basename" in p), None)
    assert problem is not None
    assert "runbook.md" in problem
    assert "domain_a/runbook.md" in problem and "domain_b/runbook.md" in problem


def test_real_sample_corpus_has_no_ambiguous_basenames() -> None:
    from rag_pipeline.evaluation.dataset import _discover_corpus, default_sample_corpus_dir

    _, problems = _discover_corpus(default_sample_corpus_dir())
    assert problems == []


# --- chunk-id leakage covers id and question ---------------------------------------


def test_chunk_id_shaped_token_in_question_rejected(
    fake_corpus: Path, answerable_record: Record
) -> None:
    case = parse_golden_case(answerable_record(question=f"What about chunk {'b' * 64}?"))
    report = validate_dataset(
        [case], sample_corpus_dir=fake_corpus, min_cases=1, min_unanswerable=0
    )
    assert any("field question contains a chunk-id-shaped token" in p for p in report.problems)


def test_chunk_id_shaped_token_in_case_id_rejected(
    fake_corpus: Path, answerable_record: Record
) -> None:
    case = parse_golden_case(answerable_record(id=f"case-{'c' * 64}"))
    report = validate_dataset(
        [case], sample_corpus_dir=fake_corpus, min_cases=1, min_unanswerable=0
    )
    assert any("field id contains a chunk-id-shaped token" in p for p in report.problems)


def test_normal_technical_identifiers_are_not_flagged_as_chunk_ids(
    fake_corpus: Path, exact_identifier_record: Record
) -> None:
    case = parse_golden_case(
        exact_identifier_record(
            question="What do ERR_DB_1042, AUTH_TOKEN_TTL, and DATABASE_POOL_TIMEOUT refer to?",
            expected_identifiers=["ERR_DB_1042", "AUTH_TOKEN_TTL", "DATABASE_POOL_TIMEOUT"],
            expected_source_files=["api-error-codes.txt"],
            notes="Mentions ERR_DB_1042 / AUTH_TOKEN_TTL / DATABASE_POOL_TIMEOUT verbatim.",
        )
    )
    report = validate_dataset(
        [case], sample_corpus_dir=fake_corpus, min_cases=1, min_unanswerable=0
    )
    assert not any("chunk-id-shaped token" in p for p in report.problems)
