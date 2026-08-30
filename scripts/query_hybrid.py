#!/usr/bin/env python3
"""Dev utility: run one hybrid (RRF-fused) query against an active index snapshot.

    question -> retrieve_dense(...) + retrieve_sparse(...)
    -> weighted Reciprocal Rank Fusion -> print top hybrid_top_k results

Uses the real `OpenAIEmbeddingProvider` for the dense channel (requires
OPENAI_API_KEY) and the real BM25 sparse channel; both need an index
already built for the chosen strategy (see scripts/index_sample_corpus.py).
It is never run automatically by the test suite. This is a development
tool, not the (not-yet-implemented) API layer.

Usage:
    python scripts/query_hybrid.py "How does authentication token expiration work?"
    python scripts/query_hybrid.py "database connection pool exhaustion" --strategy fixed
    python scripts/query_hybrid.py "webhook retry policy" --top-k 5
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from rag_pipeline.config import ChunkingStrategy, Settings
from rag_pipeline.embeddings import EmbeddingProviderError, OpenAIEmbeddingProvider
from rag_pipeline.retrieval import RetrievalError, retrieve_hybrid

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
        help="Number of fused hybrid results to return (default: settings.hybrid_top_k).",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help=(
            "Also print native dense similarity and BM25 score for each result, for "
            "debugging only -- these are never added together to produce rrf_score."
        ),
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
        results = retrieve_hybrid(
            args.question,
            strategy,
            settings,
            embedding_provider=embedding_provider,
            hybrid_top_k=args.top_k,
        )
    except RetrievalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"question: {args.question!r}")
    print(f"strategy: {strategy.value}")
    print(
        f"weights: dense={settings.rrf_dense_weight} sparse={settings.rrf_sparse_weight} "
        f"rank_constant={settings.rrf_rank_constant}"
    )
    print(f"results: {len(results)}")
    print()

    for result in results:
        location = result.source_file
        if result.section_heading:
            location += f" § {result.section_heading}"
        if result.page_number is not None:
            location += f" (page {result.page_number})"

        dense_rank = result.dense_rank if result.dense_rank is not None else "-"
        sparse_rank = result.sparse_rank if result.sparse_rank is not None else "-"

        preview = textwrap.shorten(result.text, width=_PREVIEW_CHARS, placeholder="...")
        print(
            f"[{result.rank}] rrf_score={result.rrf_score:.6f}  "
            f"dense_rank={dense_rank} sparse_rank={sparse_rank}"
        )
        print(
            f"    dense_contribution={result.dense_contribution:.6f}  "
            f"sparse_contribution={result.sparse_contribution:.6f}"
        )
        if args.diagnostics:
            similarity = (
                f"{result.dense_similarity:.4f}" if result.dense_similarity is not None else "-"
            )
            bm25 = f"{result.bm25_score:.4f}" if result.bm25_score is not None else "-"
            print(
                f"    [diagnostics] similarity={similarity} bm25={bm25} (not summed into rrf_score)"
            )
        print(f"    {location}")
        print(f"    {preview}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
