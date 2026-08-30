#!/usr/bin/env python3
"""Dev utility: ask one grounded, cited question and semantically verify its citations.

    question -> retrieve_reranked()        [dense + sparse + RRF + reranking]
    -> generate_grounded_answer()          [grounded generation with bracket citations]
    -> verify_grounded_answer()            [per-occurrence LLM-judge citation verification]
    -> print answer + citation verdicts + citation provenance map

Uses the real `OpenAIEmbeddingProvider` (dense channel), `CrossEncoderReranker`
(reranking), `OpenAIGenerator` (generation), and `OpenAICitationJudge`
(verification) -- requires `OPENAI_API_KEY` and, for reranking, the
optional `sentence-transformers` extra (`pip install 'rag-pipeline[rerank]'`).
An index must already be built for the chosen strategy (see
scripts/index_sample_corpus.py). Never run automatically by the test
suite. This is a development tool, not the (not-yet-implemented) API
layer.

Verdicts are a factual per-citation tally, not a calibrated confidence
score, and this script does not decide whether to accept, reject, or
hide the answer -- that policy is a later phase.

Never prints the API key or any hidden system/judge prompt. Rationale is
only shown with --verbose.

Usage:
    python scripts/ask_verified.py "How long do access tokens last?"
    python scripts/ask_verified.py "database connection pool exhaustion" --strategy fixed
    python scripts/ask_verified.py "webhook retry policy" --verbose
"""

from __future__ import annotations

import argparse
import sys

from rag_pipeline.config import ChunkingStrategy, Settings
from rag_pipeline.embeddings import EmbeddingProviderError, OpenAIEmbeddingProvider
from rag_pipeline.generation import (
    GenerationError,
    OpenAICitationJudge,
    OpenAIGenerator,
    resolve_citation,
    retrieve_generate_and_verify,
)
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also print each verdict's judge rationale.",
    )
    args = parser.parse_args()
    strategy = ChunkingStrategy(args.strategy)

    settings = Settings()

    try:
        embedding_provider = OpenAIEmbeddingProvider(settings)
        generator = OpenAIGenerator(settings)
        judge = OpenAICitationJudge(settings)
    except EmbeddingProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model_name,
        batch_size=settings.reranker_batch_size,
    )

    try:
        answer, report = retrieve_generate_and_verify(
            args.question,
            strategy,
            settings,
            reranker,
            generator,
            judge,
            embedding_provider=embedding_provider,
        )
    except (RetrievalError, RerankingError, GenerationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("Answer:")
    print(answer.answer_text)
    print()

    if report.total_occurrences:
        print("Citation verification:")
        for occurrence, verification in zip(report.occurrences, report.verifications, strict=True):
            print(
                f"[{verification.citation_number}] occurrence {occurrence.occurrence_id} "
                f"— {verification.verdict.value.upper()}"
            )
            if args.verbose:
                print(f"    {verification.rationale}")
        print(
            f"({report.supported_count} supported, {report.partially_supported_count} "
            f"partially supported, {report.unsupported_count} unsupported, "
            f"{report.contradicted_count} contradicted)"
        )
        print()

    if answer.cited_numbers:
        print("Sources:")
        for number in answer.cited_numbers:
            item = resolve_citation(answer.evidence, number)
            location = item.source_file
            if item.section_heading:
                location += f" — {item.section_heading}"
            if item.page_number is not None:
                location += f" (page {item.page_number})"
            print(f"[{number}] {location}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
