"""
Configuration Management System
================================
Implements environment-driven configuration with type-safe validation,
hierarchical overrides, and zero-runtime-cost abstractions through Pydantic.

Architecture: Strategy Pattern + Singleton + Functional Composition
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, Literal, Optional
from urllib.parse import urlparse, urlunparse

from pydantic import Field, PostgresDsn, RedisDsn, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """PostgreSQL configuration loaded exclusively from environment."""

    url: PostgresDsn = Field(..., alias="DATABASE_URL")
    pool_size: int = Field(default=10, ge=5, le=50, alias="DB_POOL_SIZE")
    max_overflow: int = Field(default=20, ge=5, le=100, alias="DB_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, ge=10, le=120, alias="DB_POOL_TIMEOUT")
    pool_recycle: int = Field(default=3600, ge=300, alias="DB_POOL_RECYCLE")
    echo_sql: bool = Field(default=False, alias="DB_ECHO_SQL")
    statement_timeout: int = Field(default=30000, alias="DB_STATEMENT_TIMEOUT")

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    @property
    def _parsed(self):
        return urlparse(str(self.url))

    @property
    def host(self) -> str:
        return self._parsed.hostname or ""

    @property
    def port(self) -> int:
        return self._parsed.port or 5432

    @property
    def user(self) -> str:
        return self._parsed.username or ""

    @property
    def password(self) -> Optional[SecretStr]:
        password = self._parsed.password
        return SecretStr(password) if password else None

    @property
    def database(self) -> str:
        return self._parsed.path.lstrip("/")

    @property
    def url_with_options(self) -> str:
        base = str(self.url)
        separator = "&" if "?" in base else "?"
        return f"{base}{separator}connect_timeout=10&statement_timeout={self.statement_timeout}"

    @property
    def async_url(self) -> str:
        parsed = self._parsed
        scheme = "postgresql+asyncpg"
        return urlunparse(parsed._replace(scheme=scheme))


class RedisSettings(BaseSettings):
    """Redis configuration sourced from environment variables."""

    url: RedisDsn = Field(..., alias="REDIS_URL")
    max_connections: int = Field(default=50, ge=10, le=200, alias="REDIS_MAX_CONNECTIONS")
    socket_timeout: int = Field(default=5, ge=1, le=30, alias="REDIS_SOCKET_TIMEOUT")
    socket_connect_timeout: int = Field(
        default=5, ge=1, le=30, alias="REDIS_SOCKET_CONNECT_TIMEOUT"
    )
    embedding_cache_ttl: int = Field(default=2592000, alias="REDIS_EMBEDDING_CACHE_TTL")
    llm_response_cache_ttl: int = Field(default=2592000, alias="REDIS_LLM_CACHE_TTL")
    pattern_cache_ttl: int = Field(default=604800, alias="REDIS_PATTERN_CACHE_TTL")

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    @property
    def _parsed(self):
        return urlparse(str(self.url))

    @property
    def host(self) -> str:
        return self._parsed.hostname or "localhost"

    @property
    def port(self) -> int:
        return self._parsed.port or 6379

    @property
    def db(self) -> int:
        path = self._parsed.path.lstrip("/")
        return int(path) if path else 0

    @property
    def password(self) -> Optional[SecretStr]:
        pwd = self._parsed.password
        return SecretStr(pwd) if pwd else None


class LLMSettings(BaseSettings):
    """LLM API configuration with provider fallbacks."""

    provider: str = Field(default="anthropic", alias="LLM_PROVIDER")
    anthropic_api_key: Optional[SecretStr] = Field(default=None, alias="LLM_ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-haiku-4-5-20251001", alias="LLM_ANTHROPIC_MODEL")
    openai_api_key: Optional[SecretStr] = Field(default=None, alias="LLM_OPENAI_API_KEY")
    openai_org_id: Optional[str] = Field(default=None, alias="LLM_OPENAI_ORG_ID")
    primary_model: str = Field(default="claude-haiku-4-5-20251001", alias="LLM_PRIMARY_MODEL")
    secondary_model: str = Field(default="claude-3-sonnet-20240229", alias="LLM_SECONDARY_MODEL")
    fallback_model: Optional[str] = Field(default="gpt-4-turbo-preview", alias="LLM_FALLBACK_MODEL")
    max_requests_per_minute: int = Field(
        default=50, ge=1, le=500, alias="LLM_MAX_REQUESTS_PER_MINUTE"
    )
    max_tokens_per_request: int = Field(
        default=4096, ge=100, le=128000, alias="LLM_MAX_TOKENS_PER_REQUEST"
    )
    daily_token_budget: int = Field(default=1_000_000, ge=10_000, alias="LLM_DAILY_TOKEN_BUDGET")
    cost_alert_threshold: float = Field(default=10.0, ge=0.0, alias="LLM_COST_ALERT_THRESHOLD")
    max_retries: int = Field(default=3, ge=1, le=10, alias="LLM_MAX_RETRIES")
    retry_delay: float = Field(default=1.0, ge=0.1, le=10.0, alias="LLM_RETRY_DELAY")
    default_temperature: float = Field(default=0.7, ge=0.0, le=2.0, alias="LLM_DEFAULT_TEMPERATURE")
    creative_temperature: float = Field(
        default=0.9, ge=0.0, le=2.0, alias="LLM_CREATIVE_TEMPERATURE"
    )
    deterministic_temperature: float = Field(
        default=0.1, ge=0.0, le=0.5, alias="LLM_DETERMINISTIC_TEMPERATURE"
    )
    model_pricing: Dict[str, Dict[str, float]] = Field(
        default={
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
            "claude-haiku-4-5-20251001": {"input": 0.001, "output": 0.005},
        },
        alias="LLM_MODEL_PRICING",
    )

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    @model_validator(mode="after")
    def validate_api_keys(self) -> "LLMSettings":
        if not self.anthropic_api_key and not self.openai_api_key:
            raise ValueError("Provide at least one LLM API key via environment variables.")
        return self


class NLPSettings(BaseSettings):
    """Local NLP model configuration."""

    # Sentence Transformers
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    embedding_dimension: int = Field(default=384, ge=128, le=1536)
    embedding_batch_size: int = Field(default=32, ge=1, le=256)

    # spaCy
    spacy_model: str = Field(default="en_core_web_sm")

    # Similarity thresholds
    high_similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    medium_similarity_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    low_similarity_threshold: float = Field(default=0.50, ge=0.0, le=1.0)

    # Model cache directory
    model_cache_dir: Path = Field(default=Path.home() / ".cache" / "nlp_models")

    model_config = SettingsConfigDict(env_prefix="NLP_", case_sensitive=False, extra="ignore")


class ScrapingSettings(BaseSettings):
    """Web scraping configuration."""

    user_agent: str = Field(default="ContentAutomationBot/1.0 (+http://example.com/bot)")
    request_timeout: int = Field(default=30, ge=5, le=120)
    max_retries: int = Field(default=3, ge=1, le=10)

    # Playwright settings
    headless: bool = Field(default=True)
    browser_type: Literal["chromium", "firefox", "webkit"] = Field(default="chromium")

    # Rate limiting (respectful scraping)
    min_delay_between_requests: float = Field(default=1.0, ge=0.1, le=10.0)
    max_concurrent_requests: int = Field(default=5, ge=1, le=20)

    # Content extraction
    max_article_sample_size: int = Field(default=20, ge=5, le=50)
    min_article_word_count: int = Field(default=300, ge=100)

    model_config = SettingsConfigDict(env_prefix="SCRAPING_", case_sensitive=False, extra="ignore")


class CelerySettings(BaseSettings):
    """Celery task queue configuration."""

    broker_url: str = Field(..., alias="CELERY_BROKER_URL")
    result_backend: str = Field(..., alias="CELERY_RESULT_BACKEND")

    # Worker configuration
    worker_concurrency: int = Field(default=4, ge=1, le=16)
    worker_prefetch_multiplier: int = Field(default=4, ge=1, le=10)

    # Task routing
    task_default_queue: str = Field(default="content_generation")
    task_default_routing_key: str = Field(default="content.default")

    # Timeouts
    task_soft_time_limit: int = Field(default=300, description="5 minutes")
    task_time_limit: int = Field(default=600, description="10 minutes")

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")


class TelegramSettings(BaseSettings):
    """Telegram distribution configuration."""

    bot_token: Optional[SecretStr] = Field(default=None)
    default_channel_id: Optional[str] = Field(default=None)

    # Message formatting
    parse_mode: Literal["HTML", "Markdown", "MarkdownV2"] = Field(default="HTML")
    disable_web_page_preview: bool = Field(default=False)

    model_config = SettingsConfigDict(env_prefix="TELEGRAM_", case_sensitive=False, extra="ignore")


class MonitoringSettings(BaseSettings):
    """Observability and monitoring configuration."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    log_format: Literal["json", "text"] = Field(default="json")

    # Metrics
    enable_prometheus: bool = Field(default=True)
    prometheus_port: int = Field(default=9090, ge=1024, le=65535)

    # Tracing
    enable_tracing: bool = Field(default=False)
    trace_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    
    # Security headers - CSP configuration
    enable_strict_csp: bool = Field(
        default=True,
        description="Enable strict Content-Security-Policy (disable in dev for easier debugging)"
    )
    csp_report_only: bool = Field(
        default=False,
        description="Use CSP in report-only mode (logs violations without blocking)"
    )

    model_config = SettingsConfigDict(
        env_prefix="MONITORING_", case_sensitive=False, extra="ignore"
    )


class Settings(BaseSettings):
    """
    Master configuration orchestrator.

    Implements hierarchical configuration composition with environment-specific
    overrides and runtime validation.
    """

    # Environment
    environment: Literal["development", "staging", "production"] = Field(
        default="development", alias="ENVIRONMENT"
    )
    debug: bool = Field(default=False, alias="DEBUG")

    # Application metadata
    app_name: str = Field(default="Content Automation Engine")
    app_version: str = Field(default="1.0.0")

    # Component configurations
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    nlp: NLPSettings = Field(default_factory=NLPSettings)
    scraping: ScrapingSettings = Field(default_factory=ScrapingSettings)
    celery: CelerySettings = Field(default_factory=CelerySettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)

    # Security
    secret_key: SecretStr = Field(..., alias="SECRET_KEY", description="Application secret key")
    allowed_hosts: list[str] = Field(default=["localhost", "127.0.0.1"])
    cors_origins: list[str] = Field(default=["http://localhost:3000"])
    jwt_issuer: str = Field(default="content-automation-engine")
    jwt_audience: str = Field(default="api")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("debug")
    @classmethod
    def validate_debug_mode(cls, v: bool, info) -> bool:
        """Ensure debug mode is disabled in production."""
        if info.data.get("environment") == "production" and v:
            raise ValueError("Debug mode must be disabled in production")
        return v

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: SecretStr, info) -> SecretStr:
        """Require strong, non-default secret key in production."""
        try:
            env = info.data.get("environment")
        except Exception:
            env = None

        key = v.get_secret_value() if isinstance(v, SecretStr) else str(v)
        
        # Check for weak/default secrets
        weak_secrets = [
            "change-this-to-a-secure-random-string-in-production",
            "change_me_in_production",
            "secret",
            "password",
            "12345",
        ]
        
        if key.lower() in [s.lower() for s in weak_secrets]:
            raise ValueError(
                f"SECRET_KEY cannot be a default/weak value. "
                f"Generate a secure key: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
            )
        
        if env == "production" and len(key) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters in production")
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"

    def get_log_config(self) -> dict:
        """Generate logging configuration dictionary."""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
                },
                "text": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": self.monitoring.log_format,
                    "level": self.monitoring.log_level,
                },
            },
            "root": {
                "level": self.monitoring.log_level,
                "handlers": ["console"],
            },
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Singleton factory for global settings access.

    Uses LRU cache to ensure single instance across application lifetime.
    Thread-safe and zero-overhead after first call.

    Returns:
        Settings: Validated, immutable settings instance
    """
    return Settings()


# Module-level convenience exports
settings = get_settings()

__all__ = [
    "Settings",
    "DatabaseSettings",
    "RedisSettings",
    "LLMSettings",
    "NLPSettings",
    "ScrapingSettings",
    "CelerySettings",
    "TelegramSettings",
    "MonitoringSettings",
    "get_settings",
    "settings",
]
