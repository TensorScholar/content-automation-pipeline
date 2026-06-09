"""Configuration Management - Environment-Driven Settings.

Implements type-safe configuration with:
- Environment variable loading via Pydantic BaseSettings
- Hierarchical configuration (database, Redis, LLM, etc.)
- Automatic validation and type coercion
- Singleton pattern via @lru_cache

Architecture:
    Strategy pattern with immutable settings objects.
    All configuration loaded from environment at startup.

Example:
    >>> from config.settings import get_settings
    >>> settings = get_settings()
    >>> print(settings.database.host)
    >>> print(settings.openai.api_key.get_secret_value())

Environment Variables:
    - DATABASE_URL: PostgreSQL connection string (required)
    - REDIS_URL: Redis connection string (optional)
    - OPENAI_API_KEY: OpenAI API key (optional)
    - ANTHROPIC_API_KEY: Anthropic API key (required)
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Literal, Optional
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv
from pydantic import AliasChoices, Field, RedisDsn, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

# Load environment variables from .env file
load_dotenv()


LIST_ENV_FIELDS = {"allowed_hosts", "cors_origins"}


class CommaSeparatedListEnvSource(EnvSettingsSource):
    """Support comma-separated values for selected list settings."""

    def prepare_field_value(self, field_name: str, field, value: Any, value_is_complex: bool) -> Any:
        if field_name in LIST_ENV_FIELDS and isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                return super().prepare_field_value(field_name, field, value, value_is_complex)
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return super().prepare_field_value(field_name, field, value, value_is_complex)


class CommaSeparatedListDotEnvSource(DotEnvSettingsSource):
    """Apply the same selected-list parsing to .env file values."""

    def prepare_field_value(self, field_name: str, field, value: Any, value_is_complex: bool) -> Any:
        if field_name in LIST_ENV_FIELDS and isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                return super().prepare_field_value(field_name, field, value, value_is_complex)
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return super().prepare_field_value(field_name, field, value, value_is_complex)


class DatabaseSettings(BaseSettings):
    """PostgreSQL database configuration from environment variables.

    Loads from:
        - DATABASE_URL: Primary connection string (postgresql://user:pass@host:port/db)
        - DATABASE_REPLICA_URL: Read replica connection string (defaults to DATABASE_URL)
        - DB_POOL_SIZE: Connection pool size (default: 20)
        - DB_MAX_OVERFLOW: Max overflow connections (default: 10)
        - DB_POOL_TIMEOUT: Pool checkout timeout in seconds (default: 30)
        - DB_POOL_RECYCLE: Connection recycle time in seconds (default: 3600)
        - DB_ECHO_SQL: Echo SQL queries to stdout (default: False)
        - DB_STATEMENT_TIMEOUT: Query timeout in ms (default: 30000)
    """

    url: str = Field(..., alias="DATABASE_URL")
    replica_url: Optional[str] = Field(default=None, alias="DATABASE_REPLICA_URL")
    pool_size: int = Field(default=20, ge=5, le=50, alias="DB_POOL_SIZE")
    max_overflow: int = Field(default=10, ge=5, le=100, alias="DB_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, ge=10, le=120, alias="DB_POOL_TIMEOUT")
    pool_recycle: int = Field(default=1800, ge=300, alias="DB_POOL_RECYCLE")
    echo_sql: bool = Field(default=False, alias="DB_ECHO_SQL")
    statement_timeout: int = Field(default=60000, alias="DB_STATEMENT_TIMEOUT")

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

    @property
    def async_replica_url(self) -> str:
        """Get async URL for read replica. Falls back to primary if not configured."""
        replica = self.replica_url if self.replica_url else self.url
        parsed = urlparse(str(replica))
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
    anthropic_api_key: Optional[SecretStr] = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "LLM_ANTHROPIC_API_KEY"),
    )
    anthropic_model: str = Field(default="claude-haiku-4-5-20251001", alias="LLM_ANTHROPIC_MODEL")
    gemini_api_key: Optional[SecretStr] = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY", "LLM_GEMINI_API_KEY"),
    )
    gemini_model: str = Field(default="gemini-2.5-flash-lite", alias="LLM_GEMINI_MODEL")
    # Standardized: OPENAI_API_KEY (checks both OPENAI_API_KEY and LLM_OPENAI_API_KEY for backward compat)
    openai_api_key: Optional[SecretStr] = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "LLM_OPENAI_API_KEY"),
    )
    openai_org_id: Optional[str] = Field(default=None, alias="OPENAI_ORG_ID")
    local_llm_url: Optional[str] = Field(default=None, alias="LOCAL_LLM_URL")
    primary_model: str = Field(default="claude-haiku-4-5-20251001", alias="LLM_PRIMARY_MODEL")
    secondary_model: str = Field(default="claude-haiku-4-5-20251001", alias="LLM_SECONDARY_MODEL")
    fallback_model: Optional[str] = Field(default="claude-haiku-4-5-20251001", alias="LLM_FALLBACK_MODEL")
    max_requests_per_minute: int = Field(
        default=60, ge=1, le=500, alias="LLM_MAX_REQUESTS_PER_MINUTE"
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

    # Task-specific model roles (all default to primary_model if not set)
    keyword_model: str = Field(default="claude-haiku-4-5-20251001", alias="LLM_KEYWORD_MODEL")
    planning_model: str = Field(default="claude-haiku-4-5-20251001", alias="LLM_PLANNING_MODEL")
    writing_model: str = Field(default="claude-haiku-4-5-20251001", alias="LLM_WRITING_MODEL")
    verification_model: str = Field(default="claude-haiku-4-5-20251001", alias="LLM_VERIFICATION_MODEL")
    # Single source of truth for model pricing (used by llm_client)
    model_pricing: Dict[str, Dict[str, float]] = Field(
        default={
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4o": {"input": 0.0025, "output": 0.01},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
            "claude-haiku-4-5": {"input": 0.001, "output": 0.005},
            "claude-haiku-4-5-20251001": {"input": 0.001, "output": 0.005},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
            "gemini-2.5-flash-lite": {"input": 0.0001, "output": 0.0004},
            "gemini-2.5-flash": {"input": 0.0003, "output": 0.0025},
        },
        alias="LLM_MODEL_PRICING",
    )

    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=()  # Disable model_ namespace warnings
    )

    @model_validator(mode='after')
    def validate_and_auto_detect_provider(self) -> "LLMSettings":
        """Auto-detect provider based on available API keys."""
        import os

        def configured(value: Optional[str]) -> bool:
            return bool(value and value.strip())

        # Check which API keys are available
        has_anthropic = configured(self.anthropic_api_key) or configured(os.getenv("ANTHROPIC_API_KEY"))
        has_openai = (
            configured(self.openai_api_key)
            or configured(os.getenv("OPENAI_API_KEY"))
            or configured(os.getenv("LLM_OPENAI_API_KEY"))
        )
        has_gemini = (
            configured(self.gemini_api_key)
            or configured(os.getenv("GEMINI_API_KEY"))
            or configured(os.getenv("GOOGLE_API_KEY"))
            or configured(os.getenv("LLM_GEMINI_API_KEY"))
        )
        has_local = configured(self.local_llm_url) or configured(os.getenv("LOCAL_LLM_URL"))

        # Auto-set provider if not explicitly configured
        if self.provider == "anthropic" and not has_anthropic and has_gemini:
            object.__setattr__(self, 'provider', 'gemini')
        elif self.provider == "anthropic" and not has_anthropic and has_openai:
            object.__setattr__(self, 'provider', 'openai')
        elif self.provider == "anthropic" and not has_anthropic and has_local:
            object.__setattr__(self, 'provider', 'local')
        elif self.provider == "gemini" and not has_gemini and has_anthropic:
            object.__setattr__(self, 'provider', 'anthropic')
        elif self.provider == "gemini" and not has_gemini and has_openai:
            object.__setattr__(self, 'provider', 'openai')
        elif self.provider == "gemini" and not has_gemini and has_local:
            object.__setattr__(self, 'provider', 'local')
        elif self.provider == "openai" and not has_openai and has_anthropic:
            object.__setattr__(self, 'provider', 'anthropic')
        elif self.provider == "openai" and not has_openai and has_gemini:
            object.__setattr__(self, 'provider', 'gemini')
        elif self.provider == "openai" and not has_openai and has_local:
            object.__setattr__(self, 'provider', 'local')

        provider = self.provider.lower()
        if not os.getenv("LLM_PRIMARY_MODEL"):
            if provider == "gemini":
                object.__setattr__(self, "primary_model", self.gemini_model)
            elif provider == "anthropic":
                object.__setattr__(self, "primary_model", self.anthropic_model)
            elif provider == "openai":
                object.__setattr__(self, "primary_model", os.getenv("LLM_OPENAI_MODEL", "gpt-4o-mini"))
            elif provider == "local":
                object.__setattr__(self, "primary_model", os.getenv("LOCAL_LLM_MODEL", "local-qwen-turbo"))
        if not os.getenv("LLM_SECONDARY_MODEL"):
            object.__setattr__(self, "secondary_model", self.primary_model)
        if not os.getenv("LLM_FALLBACK_MODEL"):
            object.__setattr__(self, "fallback_model", self.primary_model)

        # Log warning if no provider available
        if not has_anthropic and not has_openai and not has_gemini and not has_local:
            import logging
            logging.warning("No LLM provider configured! Set GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, or LOCAL_LLM_URL")

        if not os.getenv("LLM_KEYWORD_MODEL"):
            object.__setattr__(self, "keyword_model", self.primary_model)
        if not os.getenv("LLM_PLANNING_MODEL"):
            object.__setattr__(self, "planning_model", self.primary_model)
        if not os.getenv("LLM_WRITING_MODEL"):
            object.__setattr__(self, "writing_model", self.primary_model)
        if not os.getenv("LLM_VERIFICATION_MODEL"):
            object.__setattr__(self, "verification_model", self.primary_model)

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

    model_config = SettingsConfigDict(
        env_prefix="NLP_",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=()  # Disable model_ namespace warnings
    )


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

    # OpenTelemetry OTLP Export Configuration
    otlp_endpoint: Optional[str] = Field(
        default=None,
        alias="OTEL_EXPORTER_OTLP_ENDPOINT",
        description="OTLP exporter endpoint (e.g., http://jaeger:4317)"
    )
    otlp_service_name: str = Field(
        default="content-automation-api",
        alias="OTEL_SERVICE_NAME",
        description="Service name for traces"
    )
    otlp_insecure: bool = Field(
        default=True,
        alias="OTEL_EXPORTER_OTLP_INSECURE",
        description="Use insecure connection (no TLS) for OTLP"
    )

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
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)

    # Security
    secret_key: SecretStr = Field(..., alias="SECRET_KEY", description="Application secret key")
    allowed_hosts: list[str] = Field(default=["localhost", "127.0.0.1"], alias="ALLOWED_HOSTS")
    # CORS: Explicitly list allowed origins. Add production domain via CORS_ORIGINS env var.
    cors_origins: list[str] = Field(
        default=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3001",
        ],
        alias="CORS_ORIGINS",
        description="Allowed CORS origins (comma-separated in env var)"
    )
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM", description="JWT signing algorithm")
    jwt_issuer: str = Field(default="content-automation-engine")
    jwt_audience: str = Field(default="api")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            CommaSeparatedListEnvSource(settings_cls),
            CommaSeparatedListDotEnvSource(
                settings_cls,
                env_file=settings_cls.model_config.get("env_file"),
                env_file_encoding=settings_cls.model_config.get("env_file_encoding"),
            ),
            file_secret_settings,
        )

    @field_validator("debug")
    @classmethod
    def validate_debug_mode(cls, v: bool, info) -> bool:
        """Ensure debug mode is disabled in production."""
        if info.data.get("environment") == "production" and v:
            raise ValueError("Debug mode must be disabled in production")
        return v

    @field_validator("allowed_hosts", "cors_origins", mode="before")
    @classmethod
    def parse_string_list(cls, v: Any) -> Any:
        """Accept comma-separated env vars for list settings."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @field_validator("cors_origins", mode="after")
    @classmethod
    def validate_cors_origins(cls, v: list[str], info) -> list[str]:
        """Reject wildcard CORS origins in production."""
        try:
            env = info.data.get("environment")
        except Exception:
            env = None

        if env == "production":
            # Block wildcards and overly permissive patterns
            dangerous_patterns = ["*", "null", "file://"]
            for origin in v:
                origin_lower = origin.lower().strip()
                if any(pattern in origin_lower for pattern in dangerous_patterns):
                    raise ValueError(
                        f"Invalid CORS origin in production: '{origin}'. "
                        "Wildcards and null origins are not allowed in production environments."
                    )
                # Ensure origins are valid URLs
                if not origin.startswith(("http://", "https://")):
                    raise ValueError(
                        f"Invalid CORS origin format: '{origin}'. "
                        "Origins must start with http:// or https://"
                    )

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

        # Check for weak/default secrets - only block these in production
        weak_secrets = [
            "change-this-to-a-secure-random-string-in-production",
            "change_me_in_production",
            "secret",
            "password",
            "12345",
        ]

        if env == "production":
            if key.lower() in [s.lower() for s in weak_secrets]:
                raise ValueError(
                    f"SECRET_KEY cannot be a default/weak value in production. "
                    f"Generate a secure key: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
                )
            if len(key) < 32:
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
    "MonitoringSettings",
    "get_settings",
    "settings",
]
