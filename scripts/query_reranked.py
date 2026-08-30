#!/usr/bin/env python3
"""Dev utility: run one reranked query against an active index snapshot.

    question -> retrieve_hybrid(top_k=rerank_candidate_k)
    -> cross-encoder reranking -> print final top rerank_top_k results

Uses the real `OpenAIEmbeddingProvider` for the dense channel (requires
OPENAI_API_KEY) and the real BM25 sparse channel; both need an index
already built for the chosen strategy (see scripts/index_sample_corpus.py).
Reranking uses `CrossEncoderReranker`, which requires the optional
`sentence-transformers` extra (`pip install 'rag-pipeline[rerank]'`) and
downloads its model on first use. It is never run automatically by the
test suite. This is a development tool, not the (not-yet-implemented)
API layer.

Usage:
    python scripts/query_reranked.py "How does authentication token expiration work?"
    python scripts/query_reranked.py "database connection pool exhaustion" --strategy fixed
    python scripts/query_reranked.py "webhook retry policy" --candidate-k 30 --top-k 3
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from rag_pipeline.config import ChunkingStrategy, Settings
from rag_pipeline.embeddings import EmbeddingProviderError, OpenAIEmbeddingProvider
from rag_pipeline.reranking import CrossEncoderReranker, RerankingError
from rag_pipeline.retrieval import RetrievalError, retrieve_reranked

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
        "--candidate-k",
        type=int,
        default=None,
        help="Hybrid candidate depth fed to the reranker (default: settings.rerank_candidate_k).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of final reranked results to return (default: settings.rerank_top_k).",
    )
    args = parser.parse_args()
    strategy = ChunkingStrategy(args.strategy)

    settings = Settings()

    try:
        embedding_provider = OpenAIEmbeddingProvider(settings)
    except EmbeddingProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model_name,
        batch_size=settings.reranker_batch_size,
    )

    try:
        results = retrieve_reranked(
            args.question,
            strategy,
            settings,
            reranker,
            embedding_provider=embedding_provider,
            candidate_k=args.candidate_k,
            final_top_k=args.top_k,
        )
    except (RetrievalError, RerankingError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"question: {args.question!r}")
    print(f"strategy: {strategy.value}")
    print(f"reranker model: {settings.reranker_model_name}")
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
            f"[{result.rank}] reranker_score={result.reranker_score:.6f}  "
            f"(hybrid_rank={result.hybrid_rank} rrf_score={result.rrf_score:.6f})"
        )
        print(f"    dense_rank={dense_rank} sparse_rank={sparse_rank}")
        print(f"    {location}")
        print(f"    {preview}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
