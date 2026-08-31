"""Deterministic loading and validation of the golden evaluation dataset (Phase 4 Step 1).

`load_golden_dataset()` parses `data/eval/golden_qa.jsonl` (one JSON
object per line) into an ordered tuple of `GoldenQACase`, rejecting any
malformed record, duplicate case ID, or violated invariant loudly rather
than skipping it. `validate_dataset()` layers dataset-level and
corpus-grounding checks on top (minimum size, minimum unanswerable count,
category coverage, every expected source file really present under
`data/sample/`, exact-identifier tokens really occurring in their
sources, no chunk IDs anywhere).

No network, no API key, no model download. Nothing here runs the RAG
pipeline or consults its outputs -- golden truth comes only from the
committed corpus.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TypeVar

from .exceptions import GoldenDatasetError
from .models import Answerability, Difficulty, GoldenQACase, QuestionType

_EnumT = TypeVar("_EnumT", bound=StrEnum)

_CORPUS_EXTENSIONS = {".md", ".txt", ".html", ".pdf"}
_TEXT_EXTENSIONS = {".md", ".txt", ".html"}

# A retrieval chunk_id is a SHA-256 hex digest; golden truth must never
# pin one (see `evaluation.models`). Used to reject accidental leakage.
_CHUNK_ID_RE = re.compile(r"\b[0-9a-f]{64}\b")

_ALLOWED_KEYS = frozenset(
    {
        "id",
        "question",
        "answerability",
        "question_type",
        "difficulty",
        "requires_multi_document_reasoning",
        "expected_answer",
        "expected_facts",
        "expected_source_files",
        "expected_identifiers",
        "acceptable_source_files",
        "tags",
        "notes",
    }
)

_REQUIRED_QUESTION_TYPES: tuple[QuestionType, ...] = tuple(QuestionType)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return here.parents[3]


def default_golden_dataset_path() -> Path:
    """Absolute path to the committed golden dataset JSONL."""
    return _repo_root() / "data" / "eval" / "golden_qa.jsonl"


def default_sample_corpus_dir() -> Path:
    """Absolute path to the committed sample corpus directory."""
    return _repo_root() / "data" / "sample"


# --- per-field parsing helpers -----------------------------------------------------


def _require_str(value: object, field_name: str, source: str) -> str:
    if not isinstance(value, str):
        raise GoldenDatasetError(
            f"{source}: field {field_name!r} must be a string, got {type(value).__name__}."
        )
    return value


def _require_nonempty_str(value: object, field_name: str, source: str) -> str:
    text = _require_str(value, field_name, source)
    if not text.strip():
        raise GoldenDatasetError(f"{source}: field {field_name!r} must not be empty/whitespace.")
    return text


def _require_bool(value: object, field_name: str, source: str) -> bool:
    if not isinstance(value, bool):
        raise GoldenDatasetError(
            f"{source}: field {field_name!r} must be a boolean, got {type(value).__name__}."
        )
    return value


def _optional_str_or_none(value: object, field_name: str, source: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_str(value, field_name, source)


def _require_str_tuple(value: object, field_name: str, source: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise GoldenDatasetError(
            f"{source}: field {field_name!r} must be a list, got {type(value).__name__}."
        )
    items: list[str] = []
    for index, item in enumerate(value):
        items.append(_require_nonempty_str(item, f"{field_name}[{index}]", source))
    if len(set(items)) != len(items):
        raise GoldenDatasetError(f"{source}: field {field_name!r} contains duplicate entries.")
    return tuple(items)


def _require_enum(value: object, enum_cls: type[_EnumT], field_name: str, source: str) -> _EnumT:
    text = _require_str(value, field_name, source)
    try:
        return enum_cls(text)
    except ValueError:
        allowed = ", ".join(member.value for member in enum_cls)
        raise GoldenDatasetError(
            f"{source}: field {field_name!r} has invalid value {text!r}; allowed: {allowed}."
        ) from None


# --- single-record parsing -------------------------------------------------------


def parse_golden_case(raw: Mapping[str, object], *, source: str = "<memory>") -> GoldenQACase:
    """Validate one raw record and build a `GoldenQACase`, or raise `GoldenDatasetError`.

    Enforces: known keys only; non-empty `id`/`question`; valid enum
    values; boolean `requires_multi_document_reasoning`; string-tuple
    shapes with no duplicates; the answerability invariants (an
    answerable case needs an `expected_answer`, non-empty
    `expected_facts`, and `expected_source_files`, and is not typed
    `unanswerable_absent`; an unanswerable case must carry no
    `expected_answer`/`expected_facts`/source files, must be typed
    `unanswerable_absent`, and must not require multi-document
    reasoning); the source-count rule (a non-multi answerable case has
    EXACTLY ONE `expected_source_files` entry -- corroborating documents
    go in `acceptable_source_files`; `requires_multi_document_reasoning`
    => >= 2 `expected_source_files`; type `multi_document_reasoning` =>
    the flag is set); the `exact_identifier` rule (>= 1
    `expected_identifiers`, at least one of which appears verbatim,
    case-insensitively, in the question); and that
    `acceptable_source_files` does not repeat an `expected_source_files`
    entry.
    """
    if not isinstance(raw, Mapping):
        raise GoldenDatasetError(f"{source}: record must be a JSON object.")

    unknown = set(raw.keys()) - _ALLOWED_KEYS
    if unknown:
        raise GoldenDatasetError(
            f"{source}: unknown field(s): {', '.join(sorted(str(k) for k in unknown))}."
        )
    missing = {
        "id",
        "question",
        "answerability",
        "question_type",
        "difficulty",
        "requires_multi_document_reasoning",
    } - set(raw.keys())
    if missing:
        raise GoldenDatasetError(
            f"{source}: missing required field(s): {', '.join(sorted(missing))}."
        )

    case_id = _require_nonempty_str(raw["id"], "id", source)
    question = _require_nonempty_str(raw["question"], "question", source)
    answerability = _require_enum(raw["answerability"], Answerability, "answerability", source)
    question_type = _require_enum(raw["question_type"], QuestionType, "question_type", source)
    difficulty = _require_enum(raw["difficulty"], Difficulty, "difficulty", source)
    requires_multi_doc = _require_bool(
        raw["requires_multi_document_reasoning"], "requires_multi_document_reasoning", source
    )

    expected_answer = _optional_str_or_none(raw.get("expected_answer"), "expected_answer", source)
    expected_facts = _require_str_tuple(raw.get("expected_facts", []), "expected_facts", source)
    expected_source_files = _require_str_tuple(
        raw.get("expected_source_files", []), "expected_source_files", source
    )
    expected_identifiers = _require_str_tuple(
        raw.get("expected_identifiers", []), "expected_identifiers", source
    )
    acceptable_source_files = _require_str_tuple(
        raw.get("acceptable_source_files", []), "acceptable_source_files", source
    )
    tags = _require_str_tuple(raw.get("tags", []), "tags", source)
    notes = _optional_str_or_none(raw.get("notes"), "notes", source)

    overlap = set(acceptable_source_files) & set(expected_source_files)
    if overlap:
        raise GoldenDatasetError(
            f"{source}: {sorted(overlap)} appears in both expected_source_files and "
            "acceptable_source_files."
        )

    _check_answerability_invariants(
        source=source,
        question=question,
        answerability=answerability,
        question_type=question_type,
        requires_multi_doc=requires_multi_doc,
        expected_answer=expected_answer,
        expected_facts=expected_facts,
        expected_source_files=expected_source_files,
        expected_identifiers=expected_identifiers,
        acceptable_source_files=acceptable_source_files,
    )

    return GoldenQACase(
        id=case_id,
        question=question,
        answerability=answerability,
        question_type=question_type,
        difficulty=difficulty,
        requires_multi_document_reasoning=requires_multi_doc,
        expected_answer=expected_answer,
        expected_facts=expected_facts,
        expected_source_files=expected_source_files,
        expected_identifiers=expected_identifiers,
        acceptable_source_files=acceptable_source_files,
        tags=tags,
        notes=notes,
    )


def _check_answerability_invariants(
    *,
    source: str,
    question: str,
    answerability: Answerability,
    question_type: QuestionType,
    requires_multi_doc: bool,
    expected_answer: str | None,
    expected_facts: tuple[str, ...],
    expected_source_files: tuple[str, ...],
    expected_identifiers: tuple[str, ...],
    acceptable_source_files: tuple[str, ...],
) -> None:
    if answerability is Answerability.ANSWERABLE:
        if expected_answer is None:
            raise GoldenDatasetError(f"{source}: an answerable case requires an expected_answer.")
        if not expected_facts:
            raise GoldenDatasetError(
                f"{source}: an answerable case requires at least one expected_facts entry."
            )
        if not expected_source_files:
            raise GoldenDatasetError(
                f"{source}: an answerable case requires at least one expected_source_files entry."
            )
        if question_type is QuestionType.UNANSWERABLE_ABSENT:
            raise GoldenDatasetError(
                f"{source}: an answerable case must not use question_type "
                f"{QuestionType.UNANSWERABLE_ABSENT.value!r}."
            )
        if question_type is QuestionType.MULTI_DOCUMENT_REASONING and not requires_multi_doc:
            raise GoldenDatasetError(
                f"{source}: question_type {QuestionType.MULTI_DOCUMENT_REASONING.value!r} requires "
                "requires_multi_document_reasoning to be true."
            )
        # Source-label semantics: `expected_source_files` = documents REQUIRED for
        # the complete golden answer. A non-multi-document answerable case is,
        # by definition, fully supported by one document; anything else is a
        # corroborating citation and belongs in `acceptable_source_files`.
        if not requires_multi_doc and len(expected_source_files) != 1:
            raise GoldenDatasetError(
                f"{source}: a non-multi-document answerable case must list exactly one "
                f"expected_source_files entry (got {len(expected_source_files)}: "
                f"{list(expected_source_files)}); move any corroborating document to "
                "acceptable_source_files, or set requires_multi_document_reasoning=true if the "
                "complete answer genuinely needs more than one."
            )
        if question_type is QuestionType.EXACT_IDENTIFIER:
            if not expected_identifiers:
                raise GoldenDatasetError(
                    f"{source}: an exact_identifier case must list at least one "
                    "expected_identifiers entry."
                )
            lowered_question = question.lower()
            if not any(token.lower() in lowered_question for token in expected_identifiers):
                raise GoldenDatasetError(
                    f"{source}: an exact_identifier case must contain at least one of its "
                    f"expected_identifiers verbatim in the question; none of "
                    f"{list(expected_identifiers)} occurs in the question text."
                )
    else:
        if expected_answer is not None:
            raise GoldenDatasetError(
                f"{source}: an unanswerable case must not carry an authoritative expected_answer."
            )
        if expected_facts:
            raise GoldenDatasetError(
                f"{source}: an unanswerable case must not carry expected_facts."
            )
        if expected_source_files or acceptable_source_files:
            raise GoldenDatasetError(
                f"{source}: an unanswerable case must not claim expected/acceptable source files."
            )
        if question_type is not QuestionType.UNANSWERABLE_ABSENT:
            raise GoldenDatasetError(
                f"{source}: an unanswerable case must use question_type "
                f"{QuestionType.UNANSWERABLE_ABSENT.value!r}."
            )
        if requires_multi_doc:
            raise GoldenDatasetError(
                f"{source}: an unanswerable case must not require multi-document reasoning."
            )

    if requires_multi_doc and len(expected_source_files) < 2:
        raise GoldenDatasetError(
            f"{source}: requires_multi_document_reasoning is true but fewer than two "
            "expected_source_files are listed."
        )


# --- dataset loading -----------------------------------------------------------


def load_golden_dataset(path: Path | str | None = None) -> tuple[GoldenQACase, ...]:
    """Parse a golden-QA JSONL file into an ordered, de-duplicated tuple of cases.

    Blank lines are ignored. Each non-blank line must be a single JSON
    object. Raises `GoldenDatasetError` for a missing file, a line that
    is not valid JSON, any record that fails `parse_golden_case`, or a
    repeated case `id`. File order is preserved exactly (deterministic).
    """
    dataset_path = Path(path) if path is not None else default_golden_dataset_path()
    if not dataset_path.is_file():
        raise GoldenDatasetError(f"golden dataset file not found: {dataset_path}")

    cases: list[GoldenQACase] = []
    seen_ids: set[str] = set()
    for lineno, raw_line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        source = f"{dataset_path.name}:{lineno}"
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GoldenDatasetError(f"{source}: line is not valid JSON: {exc}") from exc
        case = parse_golden_case(obj, source=source)
        if case.id in seen_ids:
            raise GoldenDatasetError(f"{source}: duplicate case id {case.id!r}.")
        seen_ids.add(case.id)
        cases.append(case)

    return tuple(cases)


# --- dataset-level validation ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DatasetValidationReport:
    """Aggregate counts plus any problems found by `validate_dataset`.

    `problems` is empty for a valid dataset. `raise_for_problems()`
    turns a non-empty `problems` list into a single `GoldenDatasetError`.
    """

    total: int
    answerable: int
    unanswerable: int
    by_question_type: dict[str, int]
    by_difficulty: dict[str, int]
    multi_document: int
    single_document: int
    source_file_coverage: dict[str, int]
    problems: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.problems

    def raise_for_problems(self) -> None:
        if self.problems:
            joined = "\n- ".join(self.problems)
            raise GoldenDatasetError(f"golden dataset validation failed:\n- {joined}")


def _read_source_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _TEXT_EXTENSIONS:
        return path.read_text(encoding="utf-8")
    raise GoldenDatasetError(
        f"cannot read {path.name!r} as text for identifier verification (unsupported type "
        f"{suffix!r}); list a Markdown/txt/HTML source for exact-identifier cases."
    )


def validate_dataset(
    cases: Iterable[GoldenQACase],
    *,
    sample_corpus_dir: Path | str | None = None,
    min_cases: int = 50,
    min_unanswerable: int = 10,
) -> DatasetValidationReport:
    """Run dataset-level and (optionally) corpus-grounding checks; collect every problem.

    Checks: total >= `min_cases`; unanswerable >= `min_unanswerable`;
    unique IDs; every `QuestionType` represented at least once;
    per-case multi-document consistency; no chunk-id-shaped
    (64-hex) token anywhere in the golden truth, `id` and `question`
    included. When `sample_corpus_dir` is given (defaulting to
    `data/sample/`), also: the corpus contains no ambiguous duplicate
    basenames; every `expected_source_files` / `acceptable_source_files`
    entry is a real corpus basename; and every `expected_identifiers`
    entry of an `EXACT_IDENTIFIER` answerable case occurs
    (case-insensitively) in at least one of its expected sources.
    Returns a `DatasetValidationReport`; call `raise_for_problems()` to
    enforce.
    """
    case_list = list(cases)
    problems: list[str] = []

    ids = [case.id for case in case_list]
    duplicate_ids = sorted({cid for cid in ids if ids.count(cid) > 1})
    if duplicate_ids:
        problems.append(f"duplicate case id(s): {', '.join(duplicate_ids)}")

    total = len(case_list)
    answerable = sum(1 for c in case_list if c.answerability is Answerability.ANSWERABLE)
    unanswerable = total - answerable
    if total < min_cases:
        problems.append(f"dataset has {total} cases; at least {min_cases} required")
    if unanswerable < min_unanswerable:
        problems.append(
            f"dataset has {unanswerable} unanswerable cases; at least {min_unanswerable} required"
        )

    by_question_type: dict[str, int] = {qt.value: 0 for qt in QuestionType}
    by_difficulty: dict[str, int] = {d.value: 0 for d in Difficulty}
    for case in case_list:
        by_question_type[case.question_type.value] += 1
        by_difficulty[case.difficulty.value] += 1

    missing_types = [qt.value for qt in _REQUIRED_QUESTION_TYPES if by_question_type[qt.value] == 0]
    if missing_types:
        problems.append(f"no cases for question_type(s): {', '.join(missing_types)}")

    multi_document = sum(1 for c in case_list if c.requires_multi_document_reasoning)
    single_document = total - multi_document

    for case in case_list:
        if case.requires_multi_document_reasoning and len(case.expected_source_files) < 2:
            problems.append(
                f"{case.id}: requires_multi_document_reasoning but "
                f"{len(case.expected_source_files)} expected_source_files"
            )
        chunk_id_hit = _find_chunk_id_like_token(case)
        if chunk_id_hit is not None:
            problems.append(
                f"{case.id}: field {chunk_id_hit[0]} contains a chunk-id-shaped token "
                f"{chunk_id_hit[1]!r}; golden truth must not pin chunk IDs"
            )

    source_file_coverage: dict[str, int] = {}
    for case in case_list:
        for name in case.expected_source_files:
            source_file_coverage[name] = source_file_coverage.get(name, 0) + 1

    sample_dir = (
        Path(sample_corpus_dir) if sample_corpus_dir is not None else default_sample_corpus_dir()
    )
    problems.extend(_validate_against_corpus(case_list, sample_dir))

    return DatasetValidationReport(
        total=total,
        answerable=answerable,
        unanswerable=unanswerable,
        by_question_type=by_question_type,
        by_difficulty=by_difficulty,
        multi_document=multi_document,
        single_document=single_document,
        source_file_coverage=dict(sorted(source_file_coverage.items())),
        problems=tuple(problems),
    )


def _find_chunk_id_like_token(case: GoldenQACase) -> tuple[str, str] | None:
    haystacks: dict[str, Iterable[str]] = {
        "id": [case.id],
        "question": [case.question],
        "expected_answer": [case.expected_answer] if case.expected_answer else [],
        "notes": [case.notes] if case.notes else [],
        "expected_facts": case.expected_facts,
        "expected_identifiers": case.expected_identifiers,
        "expected_source_files": case.expected_source_files,
        "acceptable_source_files": case.acceptable_source_files,
        "tags": case.tags,
    }
    for field_name, values in haystacks.items():
        for value in values:
            match = _CHUNK_ID_RE.search(value)
            if match:
                return field_name, match.group(0)
    return None


def _discover_corpus(sample_dir: Path) -> tuple[dict[str, Path], list[str]]:
    """Map corpus basename -> path, and report any basename that is ambiguous.

    The benchmark uses the ingestion *basename* as source identity (that
    is exactly what `ingestion` records as `source_file`). If two files in
    different domain folders share a basename, that identity is ambiguous
    and the dataset cannot be validated against it -- so this returns a
    problem string naming the basename and every path it maps to, rather
    than silently resolving to one file. The first path (sorted) is still
    placed in the map so the remaining checks can run.
    """
    by_basename: dict[str, list[Path]] = {}
    for path in sorted(sample_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in _CORPUS_EXTENSIONS:
            by_basename.setdefault(path.name, []).append(path)

    files: dict[str, Path] = {}
    problems: list[str] = []
    for basename, paths in sorted(by_basename.items()):
        files[basename] = paths[0]
        if len(paths) > 1:
            joined = ", ".join(str(p.relative_to(sample_dir)) for p in paths)
            problems.append(
                f"ambiguous corpus basename {basename!r} maps to multiple files: {joined}"
            )
    return files, problems


def _validate_against_corpus(cases: list[GoldenQACase], sample_dir: Path) -> list[str]:
    problems: list[str] = []
    if not sample_dir.is_dir():
        return [f"sample corpus directory not found: {sample_dir}"]

    corpus, corpus_problems = _discover_corpus(sample_dir)
    problems.extend(corpus_problems)
    text_cache: dict[str, str] = {}

    for case in cases:
        for field_name in ("expected_source_files", "acceptable_source_files"):
            for name in getattr(case, field_name):
                if name not in corpus:
                    problems.append(
                        f"{case.id}: {field_name} entry {name!r} is not a file under {sample_dir}"
                    )

        if (
            case.question_type is QuestionType.EXACT_IDENTIFIER
            and case.answerability is Answerability.ANSWERABLE
        ):
            for identifier in case.expected_identifiers:
                if not _identifier_in_any_source(
                    identifier, case.expected_source_files, corpus, text_cache
                ):
                    problems.append(
                        f"{case.id}: expected identifier {identifier!r} does not occur in any of "
                        f"its expected_source_files {list(case.expected_source_files)}"
                    )
    return problems


def _identifier_in_any_source(
    identifier: str,
    source_files: tuple[str, ...],
    corpus: dict[str, Path],
    text_cache: dict[str, str],
) -> bool:
    needle = identifier.lower()
    for name in source_files:
        path = corpus.get(name)
        if path is None:
            continue
        if name not in text_cache:
            try:
                text_cache[name] = _read_source_text(path)
            except GoldenDatasetError:
                text_cache[name] = ""
        if needle in text_cache[name].lower():
            return True
    return False


def load_and_validate_golden_dataset(
    path: Path | str | None = None,
    *,
    sample_corpus_dir: Path | str | None = None,
    min_cases: int = 50,
    min_unanswerable: int = 10,
) -> tuple[GoldenQACase, ...]:
    """Load the golden dataset and raise unless it passes every validation check."""
    cases = load_golden_dataset(path)
    report = validate_dataset(
        cases,
        sample_corpus_dir=sample_corpus_dir,
        min_cases=min_cases,
        min_unanswerable=min_unanswerable,
    )
    report.raise_for_problems()
    return cases
