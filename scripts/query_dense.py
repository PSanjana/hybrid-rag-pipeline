#!/usr/bin/env python3
"""Dev utility: run one dense-retrieval query against an active index snapshot.

    question -> load active manifest for --strategy -> embed question
    -> query the active Chroma collection -> print top-k results

Uses the real `OpenAIEmbeddingProvider` (text-embedding-3-small by
default) — this makes real OpenAI API calls and requires OPENAI_API_KEY,
plus an index already built for the chosen strategy (see
scripts/index_sample_corpus.py). It is never run automatically by the test
suite. This is a development tool, not the (not-yet-implemented) API layer.

Usage:
    python scripts/query_dense.py "How does authentication token expiration work?"
    python scripts/query_dense.py "database connection pool exhaustion" --strategy fixed
    python scripts/query_dense.py "webhook retry policy" --top-k 5
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from rag_pipeline.config import ChunkingStrategy, Settings
from rag_pipeline.embeddings import EmbeddingProviderError, OpenAIEmbeddingProvider
from rag_pipeline.retrieval import RetrievalError, retrieve_dense

_PREVIEW_CHARS = 160


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("question", help="The question to search for.")
    parser.add_argument(
        "--strategy",
        choices=[s.value for s in ChunkingStrategy],
        default=ChunkingStrategy.RECURSIVE.value,
        help="Chunking strategy whose active index to query (default: recursive).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of results to return (default: settings.dense_top_k).",
    )
    args = parser.parse_args()
    strategy = ChunkingStrategy(args.strategy)

    settings = Settings()

    try:
        embedding_provider = OpenAIEmbeddingProvider(settings)
    except EmbeddingProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        results = retrieve_dense(
            args.question,
            strategy,
            settings,
            embedding_provider=embedding_provider,
            top_k=args.top_k,
        )
    except RetrievalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"question: {args.question!r}")
    print(f"strategy: {strategy.value}")
    print(f"results: {len(results)}")
    print()

    for result in results:
        location = result.source_file
        if result.section_heading:
            location += f" § {result.section_heading}"
        if result.page_number is not None:
            location += f" (page {result.page_number})"

        preview = textwrap.shorten(result.text, width=_PREVIEW_CHARS, placeholder="...")
        print(f"[{result.rank}] similarity={result.similarity:.4f} distance={result.distance:.4f}")
        print(f"    {location}")
        print(f"    {preview}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
