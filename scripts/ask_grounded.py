#!/usr/bin/env python3
"""Dev utility: ask one grounded, cited question against an active index snapshot.

    question -> retrieve_reranked() [dense + sparse + RRF + reranking]
    -> generate_grounded_answer()   [grounded generation with bracket citations]
    -> print answer + citation provenance map

Uses the real `OpenAIEmbeddingProvider` (dense channel), `CrossEncoderReranker`
(reranking), and `OpenAIGenerator` (generation) -- requires `OPENAI_API_KEY`
and, for reranking, the optional `sentence-transformers` extra
(`pip install 'rag-pipeline[rerank]'`). An index must already be built for
the chosen strategy (see scripts/index_sample_corpus.py). Never run
automatically by the test suite. This is a development tool, not the
(not-yet-implemented) API layer.

Never prints the API key or the full system/grounding prompt -- only the
final answer and a citation-to-source map.

Usage:
    python scripts/ask_grounded.py "How long do access tokens last?"
    python scripts/ask_grounded.py "database connection pool exhaustion" --strategy fixed
"""

from __future__ import annotations

import argparse
import sys

from rag_pipeline.config import ChunkingStrategy, Settings
from rag_pipeline.embeddings import EmbeddingProviderError, OpenAIEmbeddingProvider
from rag_pipeline.generation import GenerationError, OpenAIGenerator, retrieve_and_generate
from rag_pipeline.reranking import CrossEncoderReranker, RerankingError
from rag_pipeline.retrieval import RetrievalError


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("question", help="The question to ask.")
    parser.add_argument(
        "--strategy",
        choices=[s.value for s in ChunkingStrategy],
        default=ChunkingStrategy.RECURSIVE.value,
        help="Chunking strategy whose active index to query (default: recursive).",
    )
    args = parser.parse_args()
    strategy = ChunkingStrategy(args.strategy)

    settings = Settings()

    try:
        embedding_provider = OpenAIEmbeddingProvider(settings)
        generator = OpenAIGenerator(settings)
    except EmbeddingProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model_name,
        batch_size=settings.reranker_batch_size,
    )

    try:
        answer = retrieve_and_generate(
            args.question,
            strategy,
            settings,
            reranker,
            generator,
            embedding_provider=embedding_provider,
        )
    except (RetrievalError, RerankingError, GenerationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("Answer:")
    print(answer.answer_text)
    print()

    if answer.cited_numbers:
        print("Sources:")
        by_number = {item.citation_number: item for item in answer.evidence}
        for number in answer.cited_numbers:
            item = by_number[number]
            location = item.source_file
            if item.section_heading:
                location += f" — {item.section_heading}"
            if item.page_number is not None:
                location += f" (page {item.page_number})"
            print(f"[{number}] {location}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
