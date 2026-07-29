import pytest
from cryptography.fernet import Fernet
from pydantic import ValidationError

from config.settings import Settings


def set_required_settings(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://content_user:password@localhost:5432/content_automation",
    )
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("SECRET_KEY", "x" * 48)
    monkeypatch.setenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
    monkeypatch.setenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")


def test_comma_separated_security_lists_parse_from_env(monkeypatch):
    set_required_settings(monkeypatch)
    monkeypatch.setenv("ALLOWED_HOSTS", "api.example.com,localhost")
    monkeypatch.setenv("CORS_ORIGINS", "https://app.example.com,http://localhost:3001")

    settings = Settings()

    assert settings.allowed_hosts == ["api.example.com", "localhost"]
    assert settings.cors_origins == ["https://app.example.com", "http://localhost:3001"]


def test_production_selected_llm_provider_requires_matching_credentials(monkeypatch):
    set_required_settings(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("LLM_GEMINI_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("LOCAL_LLM_URL", "")

    with pytest.raises(ValidationError, match="LLM_PROVIDER=gemini requires"):
        Settings()


def test_production_rejects_primary_model_provider_mismatch(monkeypatch):
    set_required_settings(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("LLM_PRIMARY_MODEL", "local-qwen-turbo")

    with pytest.raises(ValidationError, match="LLM_PRIMARY_MODEL=.*LLM_PROVIDER=gemini"):
        Settings()


def test_production_accepts_openai_compatible_provider(monkeypatch):
    set_required_settings(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "test-compatible-key")
    monkeypatch.setenv("CREDENTIAL_ENCRYPTION_KEY", Fernet.generate_key().decode())
    for name in (
        "LLM_PRIMARY_MODEL",
        "LLM_SECONDARY_MODEL",
        "LLM_FALLBACK_MODEL",
        "LLM_KEYWORD_MODEL",
        "LLM_PLANNING_MODEL",
        "LLM_WRITING_MODEL",
        "LLM_VERIFICATION_MODEL",
    ):
        monkeypatch.setenv(name, "compatible/google/gemini-2.5-flash-lite")

    settings = Settings()

    assert settings.llm.provider == "openai_compatible"
    assert settings.llm.primary_model == "compatible/google/gemini-2.5-flash-lite"


def test_development_llm_provider_can_autodetect_available_provider(monkeypatch):
    set_required_settings(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("LLM_GEMINI_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("LLM_ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("LLM_OPENAI_API_KEY", "")

    settings = Settings()

    assert settings.llm.provider == "openai"
