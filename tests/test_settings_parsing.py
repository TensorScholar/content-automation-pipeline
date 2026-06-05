from config.settings import Settings


def test_comma_separated_security_lists_parse_from_env(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://content_user:password@localhost:5432/content_automation",
    )
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("SECRET_KEY", "x" * 48)
    monkeypatch.setenv("ALLOWED_HOSTS", "api.example.com,localhost")
    monkeypatch.setenv("CORS_ORIGINS", "https://app.example.com,http://localhost:3001")

    settings = Settings()

    assert settings.allowed_hosts == ["api.example.com", "localhost"]
    assert settings.cors_origins == ["https://app.example.com", "http://localhost:3001"]
