#!/usr/bin/env python3
"""Dev utility: run all three chunking strategies over one processed document
and print basic, non-evaluative statistics (chunk count, min/mean/max
character count). This does NOT judge which strategy is "better" — that is
a later, formal evaluation step.

Usage:
    python scripts/chunking_stats.py data/processed/<document_id>.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from rag_pipeline.chunking import Chunk, ChunkingStrategy, EmbeddingProviderError, chunk_document
from rag_pipeline.config import Settings
from rag_pipeline.ingestion.models import NormalizedDocument


def _print_stats(strategy: ChunkingStrategy, chunks: list[Chunk]) -> None:
    print(f"\n{strategy.value}:")
    if not chunks:
        print("  0 chunks")
        return
    counts = [chunk.character_count for chunk in chunks]
    print(f"  chunks: {len(chunks)}")
    print(
        f"  character_count: min={min(counts)} mean={statistics.mean(counts):.1f} max={max(counts)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_json", type=Path, help="Path to a data/processed/<id>.json file")
    args = parser.parse_args()

    raw_json = args.processed_json.read_text(encoding="utf-8")
    document = NormalizedDocument.from_dict(json.loads(raw_json))
    settings = Settings()

    print(
        f"document_id={document.document_id} source_file={document.source_file} "
        f"segments={len(document.segments)}"
    )

    for strategy in (ChunkingStrategy.FIXED, ChunkingStrategy.RECURSIVE):
        chunks = chunk_document(document, strategy=strategy, settings=settings)
        _print_stats(strategy, chunks)

    try:
        chunks = chunk_document(document, strategy=ChunkingStrategy.SEMANTIC, settings=settings)
        _print_stats(ChunkingStrategy.SEMANTIC, chunks)
    except EmbeddingProviderError as exc:
        print(f"\n{ChunkingStrategy.SEMANTIC.value}: skipped ({exc})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
