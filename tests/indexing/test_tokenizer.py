"""Tests for rag_pipeline.indexing.tokenizer."""

from rag_pipeline.indexing.tokenizer import TOKENIZER_VERSION, tokenize


def test_tokenizer_version_is_stable_identifier() -> None:
    assert TOKENIZER_VERSION == "technical_v1"


def test_normal_words_tokenize_consistently() -> None:
    assert tokenize("connection pool exhaustion") == ["connection", "pool", "exhaustion"]


def test_case_normalization() -> None:
    assert tokenize("PostgreSQL Database") == tokenize("postgresql database")
    assert tokenize("PostgreSQL") == ["postgresql"]


def test_err_auth_4017_survives_as_one_token() -> None:
    assert "err_auth_4017" in tokenize("Returns ERR_AUTH_4017 on failure.")
    assert tokenize("ERR_AUTH_4017") == ["err_auth_4017"]


def test_err_db_1042_survives_as_one_token() -> None:
    assert tokenize("ERR_DB_1042") == ["err_db_1042"]


def test_auth_token_ttl_survives_as_one_token() -> None:
    assert tokenize("AUTH_TOKEN_TTL") == ["auth_token_ttl"]


def test_database_pool_timeout_survives_as_one_token() -> None:
    assert tokenize("DATABASE_POOL_TIMEOUT") == ["database_pool_timeout"]


def test_max_webhook_retries_survives_as_one_token() -> None:
    assert tokenize("MAX_WEBHOOK_RETRIES") == ["max_webhook_retries"]


def test_deploy_freeze_survives_as_one_token() -> None:
    assert tokenize("DEPLOY_FREEZE") == ["deploy_freeze"]


def test_document_and_query_tokenizer_behavior_is_deterministic() -> None:
    text = "ERR_DB_1042: connection pool exhaustion after DATABASE_POOL_TIMEOUT."
    assert tokenize(text) == tokenize(text)
    first = tokenize(text)
    second = tokenize(text)
    assert first == second


def test_version_numbers_stay_as_one_token() -> None:
    assert tokenize("Upgrade from v1.2.3 to v2.10.0") == [
        "upgrade",
        "from",
        "v1.2.3",
        "to",
        "v2.10.0",
    ]


def test_urls_do_not_cause_pathological_tokenization() -> None:
    tokens = tokenize("See https://docs.example.org/api?ver=2 for details.")
    # Bounded, sane token count -- no giant run-on token, no crash.
    assert 1 < len(tokens) < 15
    assert all(len(token) < 50 for token in tokens)
    assert "docs" in tokens
    assert "example" in tokens


def test_no_stopword_removal() -> None:
    # "the" and "a" must survive -- this tokenizer deliberately does no
    # stopword filtering.
    assert tokenize("the pool and a connection") == ["the", "pool", "and", "a", "connection"]


def test_empty_string_returns_no_tokens() -> None:
    assert tokenize("") == []


def test_underscore_never_treated_as_separator() -> None:
    tokens = tokenize("MAX_WEBHOOK_RETRIES and DEPLOY_FREEZE together")
    assert "max_webhook_retries" in tokens
    assert "deploy_freeze" in tokens
    assert "max" not in tokens
    assert "webhook" not in tokens
