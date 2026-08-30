"""Application configuration."""

from __future__ import annotations

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
