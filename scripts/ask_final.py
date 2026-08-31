#!/usr/bin/env python3
"""Dev utility: ask one question and print the FINAL answer after the abstention policy.

    question -> retrieve_reranked()        [dense + sparse + RRF + reranking]
    -> generate_grounded_answer()          [grounded generation with bracket citations]
    -> verify_grounded_answer()            [per-occurrence LLM-judge citation verification]
    -> score_confidence()                  [deterministic heuristic confidence signal]
    -> apply_abstention_policy()           [deterministic ANSWER / abstain decision]
    -> print the user-facing FinalAnswer.answer_text (and, with --debug, the decision)

By default this prints ONLY the final user-facing answer -- either the
grounded substantive answer or the fixed graceful-abstention sentence.
The rejected draft answer is never shown in normal mode. With --debug it
also prints the decision enum, the heuristic confidence score, and the
citation-verification counts.

The confidence score is a HEURISTIC signal, NOT a calibrated probability
and NOT a "percent chance the answer is correct". The abstention
threshold is an initial uncalibrated heuristic (Phase 4 evaluation is
expected to tune it).

Uses the real `OpenAIEmbeddingProvider`, `CrossEncoderReranker`,
`OpenAIGenerator`, and `OpenAICitationJudge` -- requires `OPENAI_API_KEY`
and, for reranking, the optional `sentence-transformers` extra
(`pip install 'rag-pipeline[rerank]'`). An index must already be built
for the chosen strategy (see scripts/index_sample_corpus.py). Never run
automatically by the test suite. Never prints the API key or any hidden
system/judge prompt.

Usage:
    python scripts/ask_final.py "How long do access tokens last?"
    python scripts/ask_final.py "connection pool exhaustion" --strategy fixed
    python scripts/ask_final.py "webhook retry policy" --debug
"""

from __future__ import annotations

import argparse
import sys

from rag_pipeline.config import ChunkingStrategy, Settings
from rag_pipeline.embeddings import EmbeddingProviderError, OpenAIEmbeddingProvider
from rag_pipeline.generation import (
    AnswerDecision,
    GenerationError,
    OpenAICitationJudge,
    OpenAIGenerator,
    answer_question_with_policy,
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
        "--debug",
        "--verbose",
        dest="debug",
        action="store_true",
        help="Also print the decision, heuristic confidence score, and citation counts.",
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
        final = answer_question_with_policy(
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

    print(final.answer_text)

    if args.debug:
        confidence = final.confidence
        print()
        print("--- debug (not shown by default) ---")
        print(f"decision: {final.decision.value}")
        if final.decision is not AnswerDecision.ANSWERED:
            print(f"abstention reason: {final.abstention_reason}")
        print(
            f"heuristic confidence score: {confidence.score:.2f} "
            f"(threshold {settings.confidence_threshold:.2f}; heuristic, not a probability)"
        )
        print(
            "citation verification: "
            f"{confidence.supported_count} supported, "
            f"{confidence.partially_supported_count} partial, "
            f"{confidence.unsupported_count} unsupported, "
            f"{confidence.contradicted_count} contradicted "
            f"(of {confidence.total_citation_occurrences} occurrence(s))"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
