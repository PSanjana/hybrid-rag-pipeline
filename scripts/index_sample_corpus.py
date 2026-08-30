#!/usr/bin/env python3
"""Dev utility: index the committed sample corpus (data/sample) end-to-end.

    data/sample -> ingest -> chosen chunking strategy -> indexing service
    -> Chroma + BM25 snapshot -> print summary

Uses the real `OpenAIEmbeddingProvider` (text-embedding-3-small by
default) — this makes real OpenAI API calls and requires OPENAI_API_KEY.
It is never run automatically by the test suite.

Usage:
    python scripts/index_sample_corpus.py --strategy recursive
    python scripts/index_sample_corpus.py --strategy fixed
    python scripts/index_sample_corpus.py --strategy semantic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag_pipeline.chunking import Chunk, ChunkingStrategy, chunk_document
from rag_pipeline.config import Settings
from rag_pipeline.deduplication.models import DuplicateType
from rag_pipeline.embeddings import EmbeddingProviderError, OpenAIEmbeddingProvider
from rag_pipeline.indexing import index_chunks
from rag_pipeline.indexing.dedup_report import load_dedup_report
from rag_pipeline.indexing.sparse import load_sparse_snapshot
from rag_pipeline.ingestion import ingest_document

SAMPLE_ROOT = Path(__file__).resolve().parent.parent / "data" / "sample"
_SUPPORTED_EXTENSIONS = {".md", ".txt", ".html", ".pdf"}


def _sample_files() -> list[Path]:
    return sorted(
        f
        for f in SAMPLE_ROOT.rglob("*")
        if f.is_file() and f.suffix.lower() in _SUPPORTED_EXTENSIONS
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        choices=[s.value for s in ChunkingStrategy],
        default=ChunkingStrategy.RECURSIVE.value,
        help="Chunking strategy to use (default: recursive).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Print per-duplicate details (skipped/canonical chunk IDs, similarity, "
            "source filenames). Never prints raw chunk text."
        ),
    )
    args = parser.parse_args()
    strategy = ChunkingStrategy(args.strategy)

    settings = Settings()

    try:
        # Shared with semantic chunking (if selected) rather than
        # constructing an unrelated second client.
        embedding_provider = OpenAIEmbeddingProvider(settings)
    except EmbeddingProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    files = _sample_files()
    chunks: list[Chunk] = []
    for path in files:
        document = ingest_document(path, settings=settings)
        chunks.extend(
            chunk_document(
                document,
                strategy=strategy,
                settings=settings,
                embedding_provider=embedding_provider,
            )
        )

    try:
        result = index_chunks(chunks, settings, embedding_provider=embedding_provider)
    except EmbeddingProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    manifest = result.manifest
    sparse_snapshot = load_sparse_snapshot(settings, manifest.snapshot_id)
    dedup_report = load_dedup_report(settings, manifest.snapshot_id)
    exact_count = sum(
        1 for record in dedup_report.duplicates if record.duplicate_type == DuplicateType.EXACT
    )
    near_count = sum(
        1 for record in dedup_report.duplicates if record.duplicate_type == DuplicateType.NEAR
    )

    print(f"documents: {len(files)}")
    print(f"chunks: {len(chunks)}")
    print(f"strategy: {strategy.value}")
    print(f"embedding_model: {manifest.embedding_model}")
    print(f"embedding_dimension: {manifest.embedding_dimension}")
    print(f"reused_existing_snapshot: {result.reused_existing}")
    print(f"snapshot_id: {manifest.snapshot_id}")
    print(f"chroma_collection_name: {manifest.chroma_collection_name}")
    print(f"bm25_corpus_size: {len(sparse_snapshot.chunk_ids)}")
    print(f"pre_dedup_chunk_count: {manifest.pre_dedup_chunk_count}")
    print(f"indexed_chunk_count: {manifest.chunk_count}")
    print(f"duplicate_count: {manifest.duplicate_count}")
    print(f"exact_duplicate_count: {exact_count}")
    print(f"near_duplicate_count: {near_count}")
    print(f"dedup_similarity_threshold: {manifest.dedup_similarity_threshold}")
    print(f"dedup_report_path: {manifest.dedup_report_path}")
    print(f"chroma_dir: {settings.chroma_dir}")
    print(f"bm25_dir: {settings.bm25_dir}")
    print(f"manifest_path: {settings.manifests_dir / f'{strategy.value}.json'}")

    if args.verbose:
        for record in dedup_report.duplicates:
            print(
                f"  duplicate: skipped={record.skipped_chunk_id[:12]} "
                f"canonical={record.canonical_chunk_id[:12]} "
                f"type={record.duplicate_type.value} similarity={record.similarity:.4f} "
                f"skipped_source={record.skipped_source_file} "
                f"canonical_source={record.canonical_source_file}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
