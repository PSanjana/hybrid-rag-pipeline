#!/usr/bin/env python3
"""Dev utility: ask one grounded, cited question and print its heuristic confidence.

    question -> retrieve_reranked()        [dense + sparse + RRF + reranking]
    -> generate_grounded_answer()          [grounded generation with bracket citations]
    -> verify_grounded_answer()            [per-occurrence LLM-judge citation verification]
    -> score_confidence()                  [deterministic, decomposable confidence signal]
    -> print answer + confidence diagnostics + citation provenance map

Uses the real `OpenAIEmbeddingProvider` (dense channel), `CrossEncoderReranker`
(reranking), `OpenAIGenerator` (generation), and `OpenAICitationJudge`
(verification) -- requires `OPENAI_API_KEY` and, for reranking, the
optional `sentence-transformers` extra (`pip install 'rag-pipeline[rerank]'`).
An index must already be built for the chosen strategy (see
scripts/index_sample_corpus.py). Never run automatically by the test
suite. This is a development tool, not the (not-yet-implemented) API
layer.

The printed score is a HEURISTIC CONFIDENCE SCORE, not a calibrated
probability and not a "percent chance the answer is correct". It
combines semantic citation-support verdicts (dominant, default weight
0.9) with weak dense+sparse retrieval-channel agreement for the cited
evidence (default weight 0.1). This script does NOT decide whether to
accept, reject, or hide the answer -- that abstention policy is a later
phase (Phase 3 Step 4).

Never prints the API key or any hidden system/judge prompt. Citation
rationales are only shown with --verbose.

Usage:
    python scripts/ask_with_confidence.py "How long do access tokens last?"
    python scripts/ask_with_confidence.py "connection pool exhaustion" --strategy fixed
    python scripts/ask_with_confidence.py "webhook retry policy" --verbose
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
    retrieve_generate_verify_and_score,
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
        help="Also print each citation verdict's judge rationale.",
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
        answer, report, assessment = retrieve_generate_verify_and_score(
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

    print("Confidence diagnostics (heuristic confidence score -- NOT a calibrated probability):")
    print(f"  score: {assessment.score:.2f}")
    print(f"  citation support: {assessment.citation_support_score:.2f}")
    print(f"  retrieval agreement: {assessment.retrieval_agreement_score:.2f}")
    print(f"  supported citations: {assessment.supported_count}")
    print(f"  partial: {assessment.partially_supported_count}")
    print(f"  unsupported: {assessment.unsupported_count}")
    print(f"  contradicted: {assessment.contradicted_count}")
    print(f"  total citation occurrences: {assessment.total_citation_occurrences}")
    print(
        f"  cited evidence chunks: {assessment.unique_cited_evidence_count} "
        f"({assessment.dual_channel_cited_evidence_count} dense+sparse dual-channel)"
    )
    print(f"  has contradiction: {assessment.has_contradiction}")
    print(f"  insufficient evidence: {assessment.is_insufficient_evidence}")
    print(
        f"  component weights: citation={assessment.citation_weight:g}, "
        f"retrieval_agreement={assessment.retrieval_agreement_weight:g}"
    )
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
