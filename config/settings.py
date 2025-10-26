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

from pydantic import Field, PostgresDsn, RedisDsn, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """PostgreSQL configuration with connection pooling parameters."""

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, ge=1024, le=65535)
    user: str = Field(description="Database username")
    password: SecretStr = Field(description="Database password")
    database: str = Field(default="content_pipeline")

    # Connection pool optimization
    pool_size: int = Field(default=10, ge=5, le=50)
    max_overflow: int = Field(default=20, ge=5, le=100)
    pool_timeout: int = Field(default=30, ge=10, le=120)
    pool_recycle: int = Field(default=3600, ge=300)

    # Performance tuning
    echo_sql: bool = Field(default=False, description="Log all SQL queries")
    statement_timeout: int = Field(default=30000, description="Statement timeout in milliseconds")

    model_config = SettingsConfigDict(env_prefix="DB_", case_sensitive=False, extra="ignore")

    @property
    def url(self) -> str:
        """Construct PostgreSQL DSN with optimal parameters."""
        return (
            f"postgresql://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.database}"
            f"?connect_timeout=10"
            f"&statement_timeout={self.statement_timeout}"
        )

    @property
    def async_url(self) -> str:
        """Async PostgreSQL DSN for asyncpg driver."""
        return (
            f"postgresql+asyncpg://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.database}"
        )


class RedisSettings(BaseSettings):
    """Redis configuration for multi-tier caching."""

    host: str = Field(default="localhost")
    port: int = Field(default=6379, ge=1024, le=65535)
    password: Optional[SecretStr] = Field(default=None)
    db: int = Field(default=0, ge=0, le=15)

    # Connection pool settings
    max_connections: int = Field(default=50, ge=10, le=200)
    socket_timeout: int = Field(default=5, ge=1, le=30)
    socket_connect_timeout: int = Field(default=5, ge=1, le=30)

    # Cache TTL defaults (seconds)
    embedding_cache_ttl: int = Field(default=2592000, description="30 days")
    llm_response_cache_ttl: int = Field(default=2592000, description="30 days")
    pattern_cache_ttl: int = Field(default=604800, description="7 days")

    model_config = SettingsConfigDict(env_prefix="REDIS_", case_sensitive=False, extra="ignore")

    @property
    def url(self) -> str:
        """Construct Redis DSN."""
        auth = f":{self.password.get_secret_value()}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class LLMSettings(BaseSettings):
    """LLM API configuration with provider fallbacks."""

    # Provider selection (anthropic or openai)
    provider: str = Field(
        default="anthropic", description="LLM provider to use: 'anthropic' or 'openai'"
    )

    # Anthropic (primary provider)
    anthropic_api_key: SecretStr = Field(description="Anthropic API key")
    anthropic_model: str = Field(
        default="claude-haiku-4-5-20251001", description="Default Anthropic model to use"
    )

    # OpenAI (secondary/optional provider)
    openai_api_key: Optional[SecretStr] = Field(default=None)
    openai_org_id: Optional[str] = Field(default=None)

    # Model selection
    primary_model: str = Field(default="claude-haiku-4-5-20251001")
    secondary_model: str = Field(default="claude-3-sonnet-20240229")
    fallback_model: Optional[str] = Field(default="gpt-4-turbo-preview")

    # Rate limiting
    max_requests_per_minute: int = Field(default=50, ge=1, le=500)
    max_tokens_per_request: int = Field(default=4096, ge=100, le=128000)

    # Cost control
    daily_token_budget: int = Field(default=1_000_000, ge=10_000)
    cost_alert_threshold: float = Field(default=10.0, ge=0.0)

    # Retry configuration
    max_retries: int = Field(default=3, ge=1, le=10)
    retry_delay: float = Field(default=1.0, ge=0.1, le=10.0)

    # Temperature defaults
    default_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    creative_temperature: float = Field(default=0.9, ge=0.0, le=2.0)
    deterministic_temperature: float = Field(default=0.1, ge=0.0, le=0.5)

    # Model pricing configuration (can be overridden via environment variables)
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
        description="Model pricing per 1K tokens (input/output costs)",
    )

    model_config = SettingsConfigDict(env_prefix="LLM_", case_sensitive=False, extra="ignore")


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

    broker_url: str = Field(default="redis://localhost:6379/1")
    result_backend: str = Field(default="redis://localhost:6379/2")

    # Worker configuration
    worker_concurrency: int = Field(default=4, ge=1, le=16)
    worker_prefetch_multiplier: int = Field(default=4, ge=1, le=10)

    # Task routing
    task_default_queue: str = Field(default="content_generation")
    task_default_routing_key: str = Field(default="content.default")

    # Timeouts
    task_soft_time_limit: int = Field(default=300, description="5 minutes")
    task_time_limit: int = Field(default=600, description="10 minutes")

    model_config = SettingsConfigDict(env_prefix="CELERY_", case_sensitive=False, extra="ignore")


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
    environment: Literal["development", "staging", "production"] = Field(default="development")
    debug: bool = Field(default=False)

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
    secret_key: SecretStr = Field(description="Application secret key")
    allowed_hosts: list[str] = Field(default=["localhost", "127.0.0.1"])
    cors_origins: list[str] = Field(default=["http://localhost:3000"])

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
