# Production Deployment

This document covers the Phase 1 production path for staging and controlled
small-scale launches. It intentionally favors Docker Compose over more complex
orchestration.

## Required Services

- nginx: public entry point and TLS termination.
- frontend: Next.js app, internal Docker network only.
- api: FastAPI/Gunicorn service, internal Docker network only.
- postgres: PostgreSQL with pgvector.
- redis: Redis for cache, token blacklist, rate limiting, Celery broker, and Celery results.
- worker: Celery worker for background generation.
- celery-beat: single scheduler instance.
- flower: optional local-only Celery monitoring.

## Required Environment Variables

Production must set these values in `.env` or the deployment environment:

- `ENVIRONMENT=production`
- `DEBUG=false`
- `SERVER_NAME`
- `POSTGRES_PASSWORD`
- `REDIS_PASSWORD`
- `DATABASE_URL`
- `REDIS_URL`
- `CELERY_BROKER_URL`
- `CELERY_RESULT_BACKEND`
- `SECRET_KEY`
- `ALLOWED_HOSTS`
- `CORS_ORIGINS`
- `LLM_PROVIDER`
- The selected provider credential:
  - `GEMINI_API_KEY`, `GOOGLE_API_KEY`, or `LLM_GEMINI_API_KEY` when `LLM_PROVIDER=gemini`
  - `OPENAI_API_KEY` or `LLM_OPENAI_API_KEY` when `LLM_PROVIDER=openai`
  - `ANTHROPIC_API_KEY` or `LLM_ANTHROPIC_API_KEY` when `LLM_PROVIDER=anthropic`
  - `LOCAL_LLM_URL` when `LLM_PROVIDER=local`

Do not commit real secret values.

## Deployment Order

1. Build images:

   ```bash
   docker compose -f docker-compose.prod.yml build
   ```

2. Start PostgreSQL and Redis:

   ```bash
   docker compose -f docker-compose.prod.yml up -d postgres redis
   ```

3. Run the one-off migration. This is explicit by design; replicated API
   containers do not run Alembic during normal production startup.

   ```bash
   docker compose -f docker-compose.prod.yml --profile migrate run --rm migrate
   ```

4. Start application services:

   ```bash
   docker compose -f docker-compose.prod.yml up -d api worker celery-beat frontend nginx
   ```

5. Optional HTTPS overlay. The base compose file runs nginx in HTTP mode for
   staging and internal smoke checks. Use the HTTPS overlay only after the
   certificate files exist under `nginx/ssl`.

   ```bash
   docker compose -f docker-compose.prod.yml -f docker-compose.prod.https.yml up -d
   ```

6. Run smoke checks:

   ```bash
   docker compose -f docker-compose.prod.yml run --rm api \
     python scripts/maintenance/production_smoke_check.py \
       --api-url http://api:8000 \
       --frontend-url http://frontend:3001 \
       --nginx-url http://nginx
   ```

   Add `--check-readiness` only when the selected LLM provider key is real and
   the environment is allowed to perform provider health checks.

## Metrics

Prometheus scrapes the API container at `api:8000/metrics` over the internal
Docker network. The production nginx config blocks both `/metrics` and
`/api/metrics` publicly. The manager-facing metrics endpoint remains
`/system/metrics` and requires a superuser token.

## Rollback Basics

1. Stop the new app containers.
2. Restore the previous image tag by setting `IMAGE_TAG` to the known-good tag.
3. Start the previous services.
4. Restore the database from backup only if the failed release ran a destructive
   or incompatible migration.

Always take a database backup before running migrations:

```bash
scripts/maintenance/backup_database.sh
```

## Known Phase 1 Limitations

- WordPress credentials still need encryption-at-rest hardening.
- LLM cost accounting still needs a persistent distributed quota ledger.
- Generated exports still use local storage.
- Sentry or equivalent error tracking is not wired as a required service.
- Prompt-injection policy and moderation boundaries are still Phase 2 work.
- Full browser E2E coverage is not yet required by CI.
