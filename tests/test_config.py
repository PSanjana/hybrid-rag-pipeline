"""Tests for rag_pipeline.config."""

import os
from collections.abc import Iterator

import pytest

from rag_pipeline.config import Settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Ensure tests are isolated from the real environment and any local .env file."""
    for key in ("ENVIRONMENT", "LOG_LEVEL", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    yield


def test_settings_created_without_openai_api_key() -> None:
    settings = Settings(_env_file=None)
    assert settings.openai_api_key is None


def test_settings_default_environment_and_log_level() -> None:
    settings = Settings(_env_file=None)
    assert settings.environment == "development"
    assert settings.log_level == "INFO"


def test_settings_read_values_from_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    settings = Settings(_env_file=None)

    assert settings.environment == "production"
    assert settings.log_level == "DEBUG"
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "sk-test-key"


def test_settings_repr_does_not_leak_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret")
    settings = Settings(_env_file=None)
    assert "sk-super-secret" not in repr(settings)
    assert "sk-super-secret" not in str(settings)


def test_settings_module_import_does_not_require_env_vars() -> None:
    for key in ("ENVIRONMENT", "LOG_LEVEL", "OPENAI_API_KEY"):
        assert key not in os.environ
    Settings(_env_file=None)
