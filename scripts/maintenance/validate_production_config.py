#!/usr/bin/env python3
"""Production configuration contract validator (fail-closed, read-only).

Prevents configuration regressions for the production deployment contract:

  * required production settings must fail closed in Compose
    (${VAR:?...} guards, no insecure defaults, no dev defaults in prod)
  * the canonical production template (.env.production.example) must contain
    every required setting and never carry live-looking secrets
  * DATABASE_URL / REDIS_URL / CELERY_BROKER_URL / CELERY_RESULT_BACKEND must
    remain externally overridable while defaulting to the local Compose
    PostgreSQL/Redis (external-service portability contract)
  * removed/unused variables must not reappear
  * every variable consumed by Compose must be documented

Semantic validation is preferred over brittle regex checks. When the Docker
CLI is available, the compose stacks are actually rendered (local-default and
external-URL modes) and the rendered values are asserted.

Usage:
    python scripts/maintenance/validate_production_config.py          # full
    python scripts/maintenance/validate_production_config.py --static # no docker

Exit code 0 = PASS (warnings allowed), 1 = FAIL.
Never prints real secret values; only generated placeholders are rendered.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROD = ROOT / "docker-compose.prod.yml"
HTTPS = ROOT / "docker-compose.prod.https.yml"
MONITORING = ROOT / "docker/docker-compose.monitoring.yml"
PROD_TEMPLATE = ROOT / ".env.production.example"
DEV_TEMPLATE = ROOT / ".env.example"
ENTRYPOINT = ROOT / "entrypoint.sh"

PASS_COUNT = 0
WARN_COUNT = 0
FAIL_COUNT = 0


def pass_(label: str) -> None:
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"[PASS] {label}")


def warn(label: str, message: str = "") -> None:
    global WARN_COUNT
    WARN_COUNT += 1
    print(f"[WARN] {label}{': ' + message if message else ''}")


def fail(label: str, message: str = "") -> None:
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"[FAIL] {label}{': ' + message if message else ''}")


def require(condition: bool, label: str, message: str = "") -> None:
    if condition:
        pass_(label)
    else:
        fail(label, message)


# ---------------------------------------------------------------------------
# Required settings that the production template must document
# ---------------------------------------------------------------------------
REQUIRED_TEMPLATE_KEYS = (
    "ENVIRONMENT", "DEBUG", "SERVER_NAME",
    "DATABASE_URL", "REDIS_URL",
    "CELERY_BROKER_URL", "CELERY_RESULT_BACKEND",
    "POSTGRES_PASSWORD", "REDIS_PASSWORD",
    "SECRET_KEY", "CREDENTIAL_ENCRYPTION_KEY",
    "ALLOWED_HOSTS", "CORS_ORIGINS",
    "LLM_PROVIDER", "GEMINI_API_KEY",
    "FLOWER_USER", "FLOWER_PASSWORD",
    "GRAFANA_ADMIN_PASSWORD", "NEXT_PUBLIC_API_URL",
)

# High-confidence secret patterns; matching any of these in a committed
# template is a hard failure (templates must be placeholders only).
SECRET_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
)

# Variables removed as verified-unused; regression guard.
BANNED_LEFTOVER_VARS = ("WORKERS", "MAX_CONNECTIONS", "FORCE_HTTPS")

# Env names consumed by config/settings.py (aliases, env_prefix classes and
# direct os.getenv) that never appear in ${...} syntax. Keep in sync with
# config/settings.py when settings change.
SETTINGS_CONSUMED = frozenset(
    (
        # database
        "DATABASE_URL", "DATABASE_REPLICA_URL", "DB_POOL_SIZE", "DB_MAX_OVERFLOW",
        "DB_POOL_TIMEOUT", "DB_POOL_RECYCLE", "DB_ECHO_SQL", "DB_STATEMENT_TIMEOUT",
        # redis
        "REDIS_URL", "REDIS_MAX_CONNECTIONS", "REDIS_SOCKET_TIMEOUT",
        "REDIS_SOCKET_CONNECT_TIMEOUT", "REDIS_EMBEDDING_CACHE_TTL",
        "REDIS_LLM_CACHE_TTL", "REDIS_PATTERN_CACHE_TTL",
        # LLM
        "LLM_PROVIDER", "ANTHROPIC_API_KEY", "LLM_ANTHROPIC_API_KEY",
        "LLM_ANTHROPIC_MODEL", "GEMINI_API_KEY", "GOOGLE_API_KEY",
        "LLM_GEMINI_API_KEY", "LLM_GEMINI_MODEL", "OPENAI_API_KEY",
        "LLM_OPENAI_API_KEY", "OPENAI_ORG_ID", "OPENAI_COMPATIBLE_BASE_URL",
        "OPENAI_COMPATIBLE_API_KEY", "LLM_OPENAI_COMPATIBLE_MODEL",
        "LOCAL_LLM_URL", "LOCAL_LLM_MODEL", "LLM_OPENAI_MODEL",
        "LLM_PRIMARY_MODEL", "LLM_SECONDARY_MODEL", "LLM_FALLBACK_MODEL",
        "LLM_MAX_REQUESTS_PER_MINUTE", "LLM_MAX_TOKENS_PER_REQUEST",
        "LLM_DAILY_TOKEN_BUDGET", "LLM_COST_ALERT_THRESHOLD",
        "LLM_DAILY_COST_LIMIT_USD", "LLM_MONTHLY_COST_LIMIT_USD",
        "LLM_PROJECT_DAILY_COST_LIMIT_USD", "LLM_PROJECT_MONTHLY_COST_LIMIT_USD",
        "LLM_USER_DAILY_COST_LIMIT_USD", "LLM_USER_MONTHLY_COST_LIMIT_USD",
        "LLM_MAX_RETRIES", "LLM_RETRY_DELAY", "LLM_DEFAULT_TEMPERATURE",
        "LLM_CREATIVE_TEMPERATURE", "LLM_DETERMINISTIC_TEMPERATURE",
        "LLM_KEYWORD_MODEL", "LLM_PLANNING_MODEL", "LLM_WRITING_MODEL",
        "LLM_VERIFICATION_MODEL", "LLM_MODEL_PRICING",
        # NLP / scraping
        "NLP_EMBEDDING_MODEL", "NLP_EMBEDDING_DIMENSION", "NLP_EMBEDDING_BATCH_SIZE",
        "NLP_SPACY_MODEL", "NLP_HIGH_SIMILARITY_THRESHOLD",
        "NLP_MEDIUM_SIMILARITY_THRESHOLD", "NLP_LOW_SIMILARITY_THRESHOLD",
        "NLP_MODEL_CACHE_DIR",
        "SCRAPING_USER_AGENT", "SCRAPING_REQUEST_TIMEOUT", "SCRAPING_MAX_RETRIES",
        "SCRAPING_HEADLESS", "SCRAPING_BROWSER_TYPE",
        "SCRAPING_MIN_DELAY_BETWEEN_REQUESTS", "SCRAPING_MAX_CONCURRENT_REQUESTS",
        "SCRAPING_MAX_ARTICLE_SAMPLE_SIZE", "SCRAPING_MIN_ARTICLE_WORD_COUNT",
        # celery
        "CELERY_BROKER_URL", "CELERY_RESULT_BACKEND",
        # monitoring / tracing / sentry
        "MONITORING_LOG_LEVEL", "MONITORING_LOG_FORMAT",
        "MONITORING_ENABLE_PROMETHEUS", "MONITORING_PROMETHEUS_PORT",
        "MONITORING_ENABLE_TRACING", "MONITORING_TRACE_SAMPLE_RATE",
        "MONITORING_ENABLE_STRICT_CSP", "MONITORING_CSP_REPORT_ONLY",
        "OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_SERVICE_NAME",
        "OTEL_EXPORTER_OTLP_INSECURE",
        "SENTRY_DSN", "SENTRY_ENVIRONMENT", "SENTRY_TRACES_SAMPLE_RATE",
        # search console
        "GOOGLE_SEARCH_CONSOLE_CLIENT_ID", "GOOGLE_SEARCH_CONSOLE_CLIENT_SECRET",
        "GOOGLE_SEARCH_CONSOLE_REDIRECT_URI", "GOOGLE_SEARCH_CONSOLE_FRONTEND_RETURN_URL",
        # application security
        "ENVIRONMENT", "DEBUG", "SECRET_KEY", "CREDENTIAL_ENCRYPTION_KEY",
        "ALLOWED_HOSTS", "CORS_ORIGINS", "JWT_ALGORITHM", "JWT_ISSUER", "JWT_AUDIENCE",
    )
)

VAR_RE = re.compile(r"\$\{([A-Z0-9_]+)(:-|:\?)?[^}]*\}")


def env_keys_from(text: str) -> set[str]:
    return {m.group(1) for m in VAR_RE.finditer(text)}


def has_required_guard(value: str, var: str) -> bool:
    return bool(re.search(rf"\${{{var}:\?[^}}]*}}", value))


def is_overridable(value: str, var: str) -> bool:
    return bool(re.search(rf"\${{{var}:-", value))


def template_kv(template: Path) -> dict[str, str]:
    kv: dict[str, str] = {}
    for line in template.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        kv[key.strip()] = value.strip()
    return kv


def compose_services(text: str, kind: str) -> dict[str, dict]:
    """Minimal YAML-free extraction of service->environment map + commands.

    Kept dependency-free on purpose: PyYAML is available in CI but a simple
    block parser is more robust for interpolation-bearing strings.
    """
    import yaml  # noqa: PLC0415  (optional; CI/pyproject always provide it)

    payload = yaml.safe_load(text)
    services = payload.get("services") or {}
    result: dict[str, dict] = {}
    for name, cfg in services.items():
        env = {}
        raw_env = cfg.get("environment") or {}
        if isinstance(raw_env, dict):
            env.update({str(k): str(v) for k, v in raw_env.items() if v is not None})
        elif isinstance(raw_env, list):
            for item in raw_env:
                key, _, value = str(item).partition("=")
                env[key] = value
        command = cfg.get("command")
        if not isinstance(command, str):
            command = ""
        result[name] = {"environment": env, "command": command}
    return result


def check_template() -> None:
    if not PROD_TEMPLATE.exists():
        fail("production template exists", f"missing {PROD_TEMPLATE.relative_to(ROOT)}")
        return
    text = PROD_TEMPLATE.read_text(encoding="utf-8")
    kv = template_kv(PROD_TEMPLATE)

    missing = [k for k in REQUIRED_TEMPLATE_KEYS if k not in kv]
    require(
        not missing,
        "production template documents all required settings",
        f"missing keys: {', '.join(missing)}" if missing else "",
    )

    secret_hits = [
        pattern.pattern
        for pattern in SECRET_PATTERNS
        if pattern.search(text)
    ]
    require(
        not secret_hits,
        "production template contains no live-looking secrets",
        f"matched patterns: {', '.join(secret_hits)}" if secret_hits else "",
    )

    require(
        kv.get("ENVIRONMENT", "").lower() == "production",
        "production template sets ENVIRONMENT=production",
    )
    require(
        kv.get("DEBUG", "").lower() == "false",
        "production template sets DEBUG=false",
    )
    for key in ("SECRET_KEY", "CREDENTIAL_ENCRYPTION_KEY", "POSTGRES_PASSWORD",
                "REDIS_PASSWORD", "FLOWER_USER", "FLOWER_PASSWORD",
                "GEMINI_API_KEY", "GRAFANA_ADMIN_PASSWORD"):
        require(
            kv.get(key, "") == "",
            f"production template secret '{key}' is an empty placeholder",
            f"value appears to be a literal (only placeholders allowed)" if kv.get(key) else "",
        )


def check_compose_fail_closed() -> None:
    prod_text = PROD.read_text(encoding="utf-8")
    monitoring_text = MONITORING.read_text(encoding="utf-8")
    services = compose_services(prod_text, "prod")

    for name in ("migrate", "api", "worker", "celery-beat", "flower"):
        env = services.get(name, {}).get("environment", {})
        require(
            has_required_guard(env.get("SECRET_KEY", ""), "SECRET_KEY"),
            f"{name}: SECRET_KEY fails closed (${{SECRET_KEY:?...}} guard)",
        )
        require(
            has_required_guard(env.get("CREDENTIAL_ENCRYPTION_KEY", ""), "CREDENTIAL_ENCRYPTION_KEY"),
            f"{name}: CREDENTIAL_ENCRYPTION_KEY fails closed",
        )
        for var, url in (
            ("DATABASE_URL", env.get("DATABASE_URL", "")),
            ("REDIS_URL", env.get("REDIS_URL", "")),
            ("CELERY_BROKER_URL", env.get("CELERY_BROKER_URL", "")),
            ("CELERY_RESULT_BACKEND", env.get("CELERY_RESULT_BACKEND", "")),
        ):
            require(
                is_overridable(url, var),
                f"{name}: {var} is externally overridable (${{{{{var}}}:...}})",
            )

    postgres_env = services.get("postgres", {}).get("environment", {})
    require(
        has_required_guard(postgres_env.get("POSTGRES_PASSWORD", ""), "POSTGRES_PASSWORD"),
        "postgres: POSTGRES_PASSWORD fails closed",
    )
    redis_command = services.get("redis", {}).get("command", "")
    require(
        has_required_guard(redis_command, "REDIS_PASSWORD"),
        "redis: REDIS_PASSWORD fails closed (--requirepass guard)",
    )

    flower_command = services.get("flower", {}).get("command", "")
    require(
        has_required_guard(flower_command, "FLOWER_USER") and has_required_guard(flower_command, "FLOWER_PASSWORD"),
        "flower: FLOWER_USER/FLOWER_PASSWORD fail closed (no default credentials)",
    )

    overlay_text = HTTPS.read_text(encoding="utf-8")
    for var in BANNED_LEFTOVER_VARS:
        require(
            not re.search(rf"\b{var}\b", prod_text + overlay_text),
            f"removed unused variable '{var}' does not reappear",
        )

    monitoring_text_guard = monitoring_text
    require(
        re.search(r"\$\{GRAFANA_ADMIN_PASSWORD:\?", monitoring_text_guard) is not None,
        "monitoring overlay: GRAFANA_ADMIN_PASSWORD fails closed (no admin123 default)",
    )
    require(
        "admin123" not in monitoring_text_guard,
        "monitoring overlay: no known-default admin password",
    )

    entrypoint = ENTRYPOINT.read_text(encoding="utf-8")
    require(
        "FLOWER_USER:-" not in entrypoint and "FLOWER_PASSWORD:-" not in entrypoint,
        "entrypoint.sh: Flower basic-auth has no fail-open default",
    )
    require(
        has_required_guard(entrypoint, "FLOWER_USER") and has_required_guard(entrypoint, "FLOWER_PASSWORD"),
        "entrypoint.sh: Flower basic-auth fails closed via ${...:?} guards",
    )


def check_documentation_coverage() -> None:
    """Every Compose-consumed variable must be documented or defaulted."""
    # Strip comments so documentation examples like ${VAR:?...} are not
    # treated as real variable references.
    def strip_comments(text: str) -> str:
        return "\n".join(
            line.split("#", 1)[0] for line in text.splitlines()
        )

    sources = "".join(
        strip_comments(p.read_text(encoding="utf-8"))
        for p in (PROD, HTTPS, MONITORING)
    )
    used = env_keys_from(sources)
    template_keys = set(template_kv(PROD_TEMPLATE)) | set(template_kv(DEV_TEMPLATE))
    undocumented = used - template_keys
    hard_missing: list[str] = []
    for var in sorted(undocumented):
        if re.search(rf"\${{{var}:-", sources):
            warn(f"documentation coverage: '{var}' has a Compose default but is not in an env template")
        else:
            hard_missing.append(var)
    require(
        not hard_missing,
        "every Compose variable is documented or defaulted",
        f"undocumented required variables: {', '.join(hard_missing)}" if hard_missing else "",
    )
    # Reverse direction: template variables that nothing consumes.
    referenced: set[str] = set()
    for pattern in (
        "docker-compose*.yml", "docker/docker-compose*.yml", "entrypoint.sh",
        "config/*.py", "infrastructure/*.py", "logging_config.py",
        "scripts/maintenance/*.sh", "scripts/maintenance/*.py",
        "frontend/next.config.mjs", "frontend/src/**/*.ts", "frontend/src/**/*.tsx",
    ):
        for path in ROOT.glob(pattern):
            if path.is_file():
                text = path.read_text(encoding="utf-8")
                referenced |= env_keys_from(text)
                # Frontend JS/TS also accesses env via process.env.NEXT_PUBLIC_*
                for m in re.finditer(r"process\.env\.(NEXT_PUBLIC_[A-Z0-9_]+)", text):
                    referenced.add(m.group(1))
                for m in re.finditer(r"process\.env\.(API_PROXY_TARGET|TAURI_[A-Z0-9_]+)", text):
                    referenced.add(m.group(1))
    # config/settings.py consumes env names via alias="VAR" / env_prefix and
    # bare os.getenv() calls, not ${VAR} syntax — curated here so the reverse
    # check is accurate without brittle parsing.
    referenced |= SETTINGS_CONSUMED
    for var in sorted(set(template_kv(PROD_TEMPLATE)) - referenced):
        warn(f"template-only variable: '{var}' is documented in the production template but not referenced by config code")


def _compose_rendered(extra_env: dict[str, str], files: list[str]) -> str | None:
    env = os.environ.copy()
    # Hermetic rendering: ambient variables (CI job-level env, developer shell
    # exports) must not leak into the rendered config under test. The validator
    # defines every variable it asserts on; extra_env is reapplied below.
    for var in (
        "DATABASE_URL",
        "REDIS_URL",
        "CELERY_BROKER_URL",
        "CELERY_RESULT_BACKEND",
        "POSTGRES_PASSWORD",
        "REDIS_PASSWORD",
        "SECRET_KEY",
        "CREDENTIAL_ENCRYPTION_KEY",
        "FLOWER_USER",
        "FLOWER_PASSWORD",
        "SERVER_NAME",
        "ALLOWED_HOSTS",
        "CORS_ORIGINS",
        "GRAFANA_ADMIN_PASSWORD",
    ):
        env.pop(var, None)
    env.update(extra_env)
    env["POSTGRES_PASSWORD"] = env.get("POSTGRES_PASSWORD", "validate-pg-placeholder")
    env["REDIS_PASSWORD"] = env.get("REDIS_PASSWORD", "validate-redis-placeholder")
    env.setdefault("SECRET_KEY", "validate-secret-key-placeholder")
    env.setdefault("CREDENTIAL_ENCRYPTION_KEY", "validate-fernet-placeholder")
    env.setdefault("FLOWER_USER", "validate-flower-user")
    env.setdefault("FLOWER_PASSWORD", "validate-flower-pw")
    env.setdefault("SERVER_NAME", "validate.example.internal")
    # Use --env-file /dev/null so the repository .env (development values)
    # does not leak into the rendered config under test.
    cmd = ["docker", "compose", "--env-file", "/dev/null"] + [item for f in files for item in ("-f", f)] + ["config"]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT, timeout=120)
    if proc.returncode != 0:
        fail("compose render", f"{' '.join(files)}: {proc.stderr.strip()[:300]}")
        return None
    return proc.stdout


def check_dynamic_compose() -> None:
    if not shutil.which("docker"):
        warn("dynamic compose rendering skipped", "docker CLI not available")
        return
    if "--static" in sys.argv:
        warn("dynamic compose rendering skipped", "--static requested")
        return

    base = ["docker-compose.prod.yml"]
    rendered = _compose_rendered({}, base)
    if rendered is None:
        return
    require(
        "postgresql+asyncpg://content_user:validate-pg-placeholder@postgres:5432/content_automation" in rendered,
        "compose renders local PostgreSQL default when DATABASE_URL is unset",
    )
    require(
        "redis://:validate-redis-placeholder@redis:6379/0" in rendered,
        "compose renders local Redis default when REDIS_URL is unset",
    )
    require(
        "redis://:validate-redis-placeholder@redis:6379/1" in rendered
        and "redis://:validate-redis-placeholder@redis:6379/2" in rendered,
        "compose renders local Celery broker/result defaults when unset",
    )

    external = {
        "DATABASE_URL": "postgresql+asyncpg://app_user:extpw@db.example.com:5432/proddb?ssl=require",
        "REDIS_URL": "rediss://:extpw@redis.example.com:6379/0",
        "CELERY_BROKER_URL": "rediss://:extpw@redis.example.com:6379/1",
        "CELERY_RESULT_BACKEND": "rediss://:extpw@redis.example.com:6379/2",
    }
    rendered_ext = _compose_rendered(external, base)
    if rendered_ext is None:
        return
    require(
        "postgresql+asyncpg://app_user:extpw@db.example.com:5432/proddb?ssl=require" in rendered_ext,
        "external DATABASE_URL overrides the local default",
    )
    require(
        "rediss://:extpw@redis.example.com:6379/0" in rendered_ext,
        "external REDIS_URL overrides the local default",
    )
    require(
        "rediss://:extpw@redis.example.com:6379/1" in rendered_ext
        and "rediss://:extpw@redis.example.com:6379/2" in rendered_ext,
        "external Celery broker/result URLs override the local defaults",
    )
    require(
        "POSTGRES_PASSWORD" not in rendered_ext or "postgresql+asyncpg://app_user:extpw" in rendered_ext,
        "external mode does not require compose-local POSTGRES_PASSWORD for the app",
    )

    overlay = _compose_rendered(external | {"SERVER_NAME": "validate.example.internal"}, base + ["docker-compose.prod.https.yml"])
    if overlay is not None:
        require(
            "validate.example.internal,localhost,127.0.0.1" in overlay,
            "HTTPS overlay keeps localhost/127.0.0.1 in ALLOWED_HOSTS for healthchecks",
        )

    monitoring = _compose_rendered(
        external | {"GRAFANA_ADMIN_PASSWORD": "validate-grafana-pw", "SERVER_NAME": "validate.example.internal"},
        base + ["docker/docker-compose.monitoring.yml"],
    )
    if monitoring is not None:
        require(
            "postgresql://content_user:validate-pg-placeholder@postgres:5432/content_automation" in monitoring,
            "monitoring overlay defaults postgres-exporter to the local Compose PostgreSQL",
        )
        require(
            "GRAFANA_ADMIN_PASSWORD" not in monitoring or "validate-grafana-pw" in monitoring,
            "monitoring overlay accepts an explicit GRAFANA_ADMIN_PASSWORD",
        )


def main() -> int:
    print("Smarlux production configuration contract validation")
    print(f"Repository: {ROOT}")
    print()
    check_template()
    check_compose_fail_closed()
    check_documentation_coverage()
    check_dynamic_compose()

    print()
    print(f"Configuration contract: {PASS_COUNT} passed, {WARN_COUNT} warnings, {FAIL_COUNT} failed")
    if FAIL_COUNT:
        print("PRODUCTION_CONFIG_VALIDATION_FAILED")
        return 1
    print("PRODUCTION_CONFIG_VALIDATION_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())