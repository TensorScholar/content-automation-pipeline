# Production Configuration

Canonical production configuration contract for Smarlux Content OS.

## 1. Configuration Model

The application is configured exclusively through environment variables loaded
at startup via `config/settings.py` (Pydantic `BaseSettings`).

```
.env file  ──►  docker compose interpolation  ──►  container environment  ──►  Settings()
.env.production.example  (canonical template, placeholders only)
.env.example             (development defaults, not for production)
```

* Every production-relevant variable is classified below.
* No application code change is required to switch between local and external
  PostgreSQL / Redis — see sections 6 and 7.
* Missing required settings cause a hard startup failure (fail closed).

## 2. Required Core Settings

These must be set in the production `.env`. Startup fails without them.

| Variable | Secret | Purpose |
|---|---|---|
| `ENVIRONMENT` | no | Must be `production`. |
| `DEBUG` | no | Must be `false`. Production rejects `true`. |
| `SERVER_NAME` | no | Public hostname (e.g. `app.example.com`). Used by nginx and `ALLOWED_HOSTS`. |
| `DATABASE_URL` | no* | PostgreSQL DSN (see §6). |
| `REDIS_URL` | no* | Redis DSN (see §7). |
| `CELERY_BROKER_URL` | no* | Celery broker (Redis DB 1 by default). |
| `CELERY_RESULT_BACKEND` | no* | Celery result backend (Redis DB 2 by default). |
| `POSTGRES_PASSWORD` | yes | Password for the local Compose PostgreSQL. Required even when `DATABASE_URL` is external if the local `postgres` service is still started (it is the Compose default). |
| `REDIS_PASSWORD` | yes | Password for the local Compose Redis. Same note as above. |
| `SECRET_KEY` | yes | JWT / session signing key, ≥ 32 chars. |
| `CREDENTIAL_ENCRYPTION_KEY` | yes | Fernet key that encrypts WordPress credentials at rest. Must be a valid Fernet key. |
| `ALLOWED_HOSTS` | no | Comma-separated Host allowlist. Must include `SERVER_NAME` plus `localhost,127.0.0.1` for container-internal healthchecks. |
| `CORS_ORIGINS` | no | Comma-separated HTTPS origins. Wildcards are rejected in production. |
| `LLM_PROVIDER` | no | Active provider: `gemini` \| `anthropic` \| `openai` \| `openai_compatible` \| `local`. The selected provider's key must be present. |
| *key for selected provider* | yes | `GEMINI_API_KEY` / `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `OPENAI_COMPATIBLE_*` / `LOCAL_LLM_URL` — whichever matches `LLM_PROVIDER`. |
| `FLOWER_USER` / `FLOWER_PASSWORD` | yes | Basic-auth for the Flower UI. No default is accepted. |
| `GRAFANA_ADMIN_PASSWORD` | yes | Grafana admin password (monitoring overlay). No default is accepted. |
| `NEXT_PUBLIC_API_URL` | no | Frontend API base. Must be `/api` in production (nginx proxy). |

\* DSN values contain embedded passwords. Treat them as secrets even though the
field itself is classified non-secret for URL handling.

Optional tuning with safe defaults: `GUNICORN_WORKERS`, `CELERY_CONCURRENCY`,
`LOG_LEVEL` / `MONITORING_LOG_LEVEL`, `IMAGE_TAG`, `LLM_*` cost and model
settings, `GRAFANA_ADMIN_USER`, `BACKUP_DIR` / `RETENTION_DAYS`.

## 3. Optional Integrations

These are disabled when their variables are empty. No startup failure.

* **Google Search Console** — all four must be set together:
  `GOOGLE_SEARCH_CONSOLE_CLIENT_ID`, `GOOGLE_SEARCH_CONSOLE_CLIENT_SECRET`,
  `GOOGLE_SEARCH_CONSOLE_REDIRECT_URI`, `GOOGLE_SEARCH_CONSOLE_FRONTEND_RETURN_URL`.
  In production both URLs must be `https://`.

* **Sentry** — `SENTRY_DSN` (empty = disabled), `SENTRY_ENVIRONMENT`,
  `SENTRY_TRACES_SAMPLE_RATE`.

* **OpenTelemetry** — `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`,
  `OTEL_EXPORTER_OTLP_INSECURE`.

* **Monitoring exporters** — `POSTGRES_EXPORTER_DSN`, `REDIS_EXPORTER_ADDR`
  (see §6 / §7).

* **Grafana** — `GRAFANA_SERVER_NAME` overrides `SERVER_NAME` for the Grafana
  root URL.

* **LLM optional providers** — any provider key not matching `LLM_PROVIDER` is
  an inert fallback. Leave empty if unused.

## 4. Secret Ownership Expectations

| Secret | Owner | Rotation |
|---|---|---|
| `SECRET_KEY` | Operator | Rotate with planned downtime (invalidates JWTs). |
| `CREDENTIAL_ENCRYPTION_KEY` | Operator | **Never rotate casually** — requires re-encrypting every stored WordPress credential. |
| `POSTGRES_PASSWORD` / `REDIS_PASSWORD` | Operator | Rotate via Compose secret update + rolling restart. |
| `FLOWER_USER` / `FLOWER_PASSWORD` | Operator | Rotate at will. |
| `GRAFANA_ADMIN_PASSWORD` | Operator | Rotate at will. |
| LLM provider keys | Operator (from provider console) | Rotate at the provider, then update `.env`. |
| Search Console client secret | Operator (Google Cloud Console) | Rotate at Google, then update `.env`. |

No secret is ever committed, logged, or printed by validation scripts. CI
validates the *shape* of the compose config, never the values.

## 5. Production Credential Issuance Rules

1. All production secrets are generated **on the production host** or in a
   secret manager. Never reuse the development `.env`.
2. Generate `SECRET_KEY`: `python -c "import secrets; print(secrets.token_urlsafe(48))"`
3. Generate `CREDENTIAL_ENCRYPTION_KEY`: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
4. Generate passwords with `openssl rand -base64 24` or equivalent.
5. Store the resulting `.env` with `chmod 600` and restrict host access.
6. Use Docker secrets or a vault for teams; the Compose files read plain env
   vars, so any secret-delivery mechanism that populates the container
   environment is compatible.

## 6. External PostgreSQL Configuration

The Compose stack resolves database URLs with nested interpolation:

```
DATABASE_URL=${DATABASE_URL:-postgresql+asyncpg://content_user:${POSTGRES_PASSWORD:?required}@postgres:5432/content_automation}
```

* **Local mode** (default) — leave `DATABASE_URL` empty. The default above is
  used and the Compose `postgres` service provides the database.
* **External mode** — set `DATABASE_URL` to a fully-qualified asyncpg DSN:
  ```
  DATABASE_URL=postgresql+asyncpg://app_user:<password>@db.example.com:5432/proddb?ssl=require
  ```
  When `DATABASE_URL` is set, the embedded `${POSTGRES_PASSWORD:?...}` is not
  evaluated — no local password is required for the application. TLS is
  enabled with `?ssl=require` (asyncpg). Alembic (`alembic/env.py`) rewrites
  this to `?sslmode=require` for the synchronous `psycopg2` migration path
  automatically.

* `POSTGRES_EXPORTER_DSN` — override when PostgreSQL is external; otherwise the
  exporter defaults to the local Compose PostgreSQL.

* The Compose `postgres` service still starts by default. External mode keeps
  it running harmlessly (localhost-only port) or it can be stopped after
  `docker compose up` with `docker compose stop postgres` if the operator
  prefers. No application code change is needed.

## 7. External Redis Configuration

Same pattern as PostgreSQL:

```
REDIS_URL=${REDIS_URL:-redis://:${REDIS_PASSWORD:?required}@redis:6379/0}
CELERY_BROKER_URL=${CELERY_BROKER_URL:-redis://:${REDIS_PASSWORD:?required}@redis:6379/1}
CELERY_RESULT_BACKEND=${CELERY_RESULT_BACKEND:-redis://:${REDIS_PASSWORD:?required}@redis:6379/2}
```

* **Local mode** — leave all three empty. Each defaults to the Compose `redis`
  service on DB 0 / 1 / 2.
* **External mode** — set all three to the provider's endpoints:
  ```
  REDIS_URL=rediss://:<password>@redis.example.com:6379/0
  CELERY_BROKER_URL=rediss://:<password>@redis.example.com:6379/1
  CELERY_RESULT_BACKEND=rediss://:<password>@redis.example.com:6379/2
  ```
  `rediss://` enables TLS. When these are set, the embedded
  `${REDIS_PASSWORD:?...}` defaults are not evaluated.

* `REDIS_EXPORTER_ADDR` / `REDIS_PASSWORD` — override when Redis is external;
  otherwise the exporter defaults to `redis://redis:6379`.

* Same note as PostgreSQL: the Compose `redis` service still starts by default
  but is unused when external URLs are set.

## 8. Frontend Public-Variable Boundary

* Only variables prefixed `NEXT_PUBLIC_` are inlined into the browser bundle
  at **build time**. They are visible to anyone who loads the page.
* **Never** put secrets, passwords, tokens, or internal hostnames behind a
  `NEXT_PUBLIC_` name.
* Production contract:
  * `NEXT_PUBLIC_API_URL=/api` — relative, routed by nginx (`/api/` → `api:8000`).
  * `NEXT_PUBLIC_GRAFANA_URL` — optional plain HTTPS URL for the Monitoring
    panel's Grafana link. Do not embed credentials or internal addresses.
  * `API_PROXY_TARGET` (`http://api:8000` inside Compose, `http://127.0.0.1:8000`
    for local dev) and `TAURI_STATIC_EXPORT` are build/runtime-only and never
    exposed to the browser as `NEXT_PUBLIC_` vars.

## 9. Environment Creation Procedure

On the **production host** (never on a developer machine):

```bash
cp .env.production.example .env
# edit .env — fill every empty value (see §2)
chmod 600 .env

# validate without starting anything
python scripts/maintenance/validate_production_config.py
docker compose --env-file /dev/null -f docker-compose.prod.yml config --quiet
docker compose --env-file /dev/null -f docker-compose.prod.yml -f docker-compose.prod.https.yml config --quiet

# start
docker compose -f docker-compose.prod.yml up -d postgres redis
docker compose -f docker-compose.prod.yml --profile migrate run --rm migrate
docker compose -f docker-compose.prod.yml up -d api worker celery-beat frontend nginx
# HTTPS overlay (after certs exist under nginx/ssl/live/<SERVER_NAME>/)
docker compose -f docker-compose.prod.yml -f docker-compose.prod.https.yml up -d
```

## 10. Validation Procedure

Three layers, from cheapest to most complete:

1. **Static contract** — no Docker required:
   ```bash
   python scripts/maintenance/validate_production_config.py --static
   ```
   Checks the production template, fail-closed guards, external-URL
   overridability, banned leftovers, and documentation coverage.

2. **Rendered contract** — Docker required, no containers started:
   ```bash
   python scripts/maintenance/validate_production_config.py
   ```
   Additionally renders the Compose stacks with local defaults and with
   external-style URLs and asserts the rendered values.

3. **Full gate** — Docker + Python + frontend toolchain:
   ```bash
   scripts/maintenance/p0_release_gate.sh --static-only   # includes the static contract
   scripts/maintenance/p0_release_gate.sh --full          # + focused P0 tests + compose interpolation
   ```
   CI runs the rendered contract in `docker-build-validate` and the full gate
   on `main`.

## 11. Local Developer `.env` Must Never Be Copied to Production

The repository-local `.env` is a **development-only** file. It may contain
weak passwords, placeholder secrets, and provider keys that have already been
exposed to local logs, shell history, or backups. Copying it to a production
host would:

* deploy known-weak credentials,
* leak development provider keys into production traffic and cost accounting,
* and invalidate the fail-closed guarantees validated by the production
  contract.

Always create the production `.env` from `.env.production.example` on the
production host as described in §9.

## See Also

* `.env.production.example` — canonical template (placeholders only).
* `.env.example` — development defaults.
* `docs/production-deployment.md` — deployment order, smoke checks, rollback.
* `scripts/maintenance/validate_production_config.py` — automated contract.
