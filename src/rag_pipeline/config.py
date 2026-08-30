"""Application configuration."""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    def __repr__(self) -> str:
        return f"Settings(environment={self.environment!r}, log_level={self.log_level!r})"


def get_settings() -> Settings:
    """Create a fresh Settings instance from the current environment."""
    return Settings()
