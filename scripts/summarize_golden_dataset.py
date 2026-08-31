#!/usr/bin/env python3
"""Print statistics about the golden evaluation dataset (Phase 4 Step 1).

Benchmark-inspection only. This script does NOT run the RAG pipeline,
does not need an API key or network, and never touches retrieval,
generation, verification, confidence, or the abstention policy. It just
loads `data/eval/golden_qa.jsonl`, validates it, and prints counts.

Usage:
    python scripts/summarize_golden_dataset.py
    python scripts/summarize_golden_dataset.py --path some/other/golden_qa.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag_pipeline.evaluation import (
    Answerability,
    GoldenDatasetError,
    default_golden_dataset_path,
    load_golden_dataset,
    validate_dataset,
)


def _bar(count: int, total: int, width: int = 24) -> str:
    filled = 0 if total == 0 else round(width * count / total)
    return "#" * filled + "-" * (width - filled)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=default_golden_dataset_path(),
        help="Path to the golden JSONL (default: data/eval/golden_qa.jsonl).",
    )
    args = parser.parse_args()

    try:
        cases = load_golden_dataset(args.path)
    except GoldenDatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    report = validate_dataset(cases)
    total = report.total

    print(f"Golden evaluation dataset: {args.path}")
    print(f"  total cases              : {total}")
    print(
        f"  answerable / unanswerable: {report.answerable} / {report.unanswerable} "
        f"({0 if total == 0 else round(100 * report.unanswerable / total)}% negative)"
    )
    print(f"  single- / multi-document : {report.single_document} / {report.multi_document}")

    print("\n  by question_type:")
    for name, count in report.by_question_type.items():
        print(f"    {name:<26} {count:>3}  {_bar(count, total)}")

    print("\n  by difficulty:")
    for name, count in report.by_difficulty.items():
        print(f"    {name:<26} {count:>3}  {_bar(count, total)}")

    print("\n  expected_source_files coverage:")
    for name, count in report.source_file_coverage.items():
        print(f"    {name:<28} {count:>3}")

    answerable_ids = sum(
        1 for c in cases if c.answerability is Answerability.ANSWERABLE and c.expected_identifiers
    )
    print(f"\n  answerable cases naming >=1 identifier: {answerable_ids}")

    if report.problems:
        print("\n  VALIDATION PROBLEMS:")
        for problem in report.problems:
            print(f"    - {problem}")
        return 1

    print("\n  validation: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
