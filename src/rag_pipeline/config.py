"""Application configuration."""

from __future__ import annotations

import math
from enum import StrEnum
from pathlib import Path

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChunkingStrategy(StrEnum):
    """Switchable chunking strategies (see `rag_pipeline.chunking`).

    Defined here rather than in `rag_pipeline.chunking` so that `config`
    stays a dependency-free leaf module: `chunking` depends on `config`,
    never the other way around.
    """

    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class Settings(BaseSettings):
    """Foundational application settings, loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = "development"
    log_level: str = "INFO"
    openai_api_key: SecretStr | None = None
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")

    chunk_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "text-embedding-3-small"
    chunk_semantic_similarity_threshold: float = 0.5

    index_root_dir: Path = Path("data/indexes")
    chroma_collection_prefix: str = "rag"
    index_batch_size: int = 500
    dedup_similarity_threshold: float = 0.95

    dense_top_k: int = 10
    sparse_top_k: int = 10

    rrf_dense_weight: float = 0.7
    rrf_sparse_weight: float = 0.3
    rrf_rank_constant: int = 60
    hybrid_top_k: int = 10

    rerank_candidate_k: int = 20
    rerank_top_k: int = 5
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_batch_size: int = 32

    generation_model: str = "gpt-5.6-terra"
    # Deliberately the same default as generation_model for now (a single
    # deliberate choice, not an accidental duplication) -- kept as its own
    # setting so the judge model can be tuned independently later.
    citation_judge_model: str = "gpt-5.6-terra"

    confidence_citation_weight: float = 0.9
    confidence_retrieval_agreement_weight: float = 0.1

    # Phase 3 Step 4 abstention policy. An initial UNCALIBRATED heuristic
    # cut-off: an answer whose deterministic ConfidenceAssessment.score is
    # below this (and which no stronger rule -- insufficiency, contradiction,
    # unsupported citation -- already rejected) is replaced by the canonical
    # "I don't know" response. Phase 4 evaluation is expected to tune this;
    # it is deliberately a separate policy setting, never reusing the Step 3
    # confidence_* component weights.
    confidence_threshold: float = 0.8

    # Phase 4 Step 2 evaluation metrics. The semantic correctness/faithfulness
    # judges use their own model setting so it can be pinned or upgraded
    # independently of generation/citation judging later. Same default as
    # the others for now -- a single deliberate choice, not a duplication.
    evaluation_judge_model: str = "gpt-5.6-terra"

    @model_validator(mode="after")
    def _validate_chunking_config(self) -> Settings:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size.")
        if not -1.0 <= self.chunk_semantic_similarity_threshold <= 1.0:
            raise ValueError("chunk_semantic_similarity_threshold must be between -1.0 and 1.0.")
        if self.index_batch_size <= 0:
            raise ValueError("index_batch_size must be positive.")
        return self

    @model_validator(mode="after")
    def _validate_dedup_config(self) -> Settings:
        if not 0.0 <= self.dedup_similarity_threshold <= 1.0:
            raise ValueError("dedup_similarity_threshold must be between 0.0 and 1.0.")
        return self

    @model_validator(mode="after")
    def _validate_retrieval_config(self) -> Settings:
        if self.dense_top_k <= 0:
            raise ValueError("dense_top_k must be positive.")
        if self.sparse_top_k <= 0:
            raise ValueError("sparse_top_k must be positive.")
        return self

    @model_validator(mode="after")
    def _validate_hybrid_config(self) -> Settings:
        if self.rrf_dense_weight < 0:
            raise ValueError("rrf_dense_weight must not be negative.")
        if self.rrf_sparse_weight < 0:
            raise ValueError("rrf_sparse_weight must not be negative.")
        if self.rrf_dense_weight + self.rrf_sparse_weight <= 0:
            raise ValueError("At least one of rrf_dense_weight/rrf_sparse_weight must be positive.")
        if self.rrf_rank_constant < 0:
            raise ValueError("rrf_rank_constant must not be negative.")
        if self.hybrid_top_k <= 0:
            raise ValueError("hybrid_top_k must be positive.")
        return self

    @model_validator(mode="after")
    def _validate_reranking_config(self) -> Settings:
        if self.rerank_candidate_k <= 0:
            raise ValueError("rerank_candidate_k must be positive.")
        if self.rerank_top_k <= 0:
            raise ValueError("rerank_top_k must be positive.")
        if self.rerank_top_k > self.rerank_candidate_k:
            raise ValueError("rerank_top_k must not exceed rerank_candidate_k.")
        if self.reranker_batch_size <= 0:
            raise ValueError("reranker_batch_size must be positive.")
        if not self.reranker_model_name.strip():
            raise ValueError("reranker_model_name must not be empty.")
        return self

    @model_validator(mode="after")
    def _validate_generation_config(self) -> Settings:
        if not self.generation_model.strip():
            raise ValueError("generation_model must not be empty.")
        if not self.citation_judge_model.strip():
            raise ValueError("citation_judge_model must not be empty.")
        return self

    @model_validator(mode="after")
    def _validate_evaluation_config(self) -> Settings:
        if not self.evaluation_judge_model.strip():
            raise ValueError("evaluation_judge_model must not be empty.")
        return self

    @model_validator(mode="after")
    def _validate_confidence_config(self) -> Settings:
        # Finiteness is checked first: NaN/+inf/-inf would slip past the
        # `< 0` and `sum <= 0` comparisons below (all of which are False
        # for NaN), and a non-finite weight would break the `[0, 1]`
        # guarantee documented on ConfidenceAssessment.
        if not math.isfinite(self.confidence_citation_weight):
            raise ValueError("confidence_citation_weight must be finite (no NaN/inf).")
        if not math.isfinite(self.confidence_retrieval_agreement_weight):
            raise ValueError("confidence_retrieval_agreement_weight must be finite (no NaN/inf).")
        if self.confidence_citation_weight < 0:
            raise ValueError("confidence_citation_weight must not be negative.")
        if self.confidence_retrieval_agreement_weight < 0:
            raise ValueError("confidence_retrieval_agreement_weight must not be negative.")
        if self.confidence_citation_weight + self.confidence_retrieval_agreement_weight <= 0:
            raise ValueError(
                "At least one of confidence_citation_weight/"
                "confidence_retrieval_agreement_weight must be positive."
            )
        return self

    @model_validator(mode="after")
    def _validate_abstention_config(self) -> Settings:
        if not math.isfinite(self.confidence_threshold):
            raise ValueError("confidence_threshold must be finite (no NaN/inf).")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0 inclusive.")
        return self

    @property
    def chroma_dir(self) -> Path:
        """Chroma's persistence directory, derived from `index_root_dir`."""
        return self.index_root_dir / "chroma"

    @property
    def bm25_dir(self) -> Path:
        """BM25 sparse-snapshot persistence directory, derived from `index_root_dir`."""
        return self.index_root_dir / "bm25"

    @property
    def manifests_dir(self) -> Path:
        """Per-strategy active-manifest directory, derived from `index_root_dir`."""
        return self.index_root_dir / "manifests"

    def __repr__(self) -> str:
        return f"Settings(environment={self.environment!r}, log_level={self.log_level!r})"


def get_settings() -> Settings:
    """Create a fresh Settings instance from the current environment."""
    return Settings()
