# Production Deployment

This document covers the Phase 1 production path for staging and controlled
small-scale launches. It intentionally favors Docker Compose over more complex
orchestration.

> **Configuration contract**: the authoritative list of required and optional
> settings, secret-issuance rules, external-database portability, and the
> validation procedure lives in [docs/production-configuration.md](production-configuration.md).
> Create the production `.env` from `.env.production.example` and validate it
> with `scripts/maintenance/validate_production_config.py` before deploying.
> Never copy the development `.env` to production.

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

Create the production `.env` from `.env.production.example`
(see [docs/production-configuration.md](production-configuration.md) for the
full contract). Summary:

- `ENVIRONMENT=production`, `DEBUG=false`, `SERVER_NAME`
- `POSTGRES_PASSWORD`, `REDIS_PASSWORD`
- `DATABASE_URL`, `REDIS_URL`, `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`
  (each is `${VAR:-local-default}` — leave empty for local Compose services,
  or set to an external DSN with `?ssl=require` / `rediss://` for managed
  PostgreSQL / Redis; see `docs/production-configuration.md` §6 / §7)
- `SECRET_KEY`, `CREDENTIAL_ENCRYPTION_KEY`
- `ALLOWED_HOSTS` (must include `SERVER_NAME,localhost,127.0.0.1` for healthchecks),
  `CORS_ORIGINS` (explicit HTTPS origins, no wildcards)
- `LLM_PROVIDER` and the selected provider credential:
  - `GEMINI_API_KEY`, `GOOGLE_API_KEY`, or `LLM_GEMINI_API_KEY` when `LLM_PROVIDER=gemini`
  - `OPENAI_API_KEY` or `LLM_OPENAI_API_KEY` when `LLM_PROVIDER=openai`
  - `ANTHROPIC_API_KEY` or `LLM_ANTHROPIC_API_KEY` when `LLM_PROVIDER=anthropic`
  - `LOCAL_LLM_URL` when `LLM_PROVIDER=local`
- `LLM_DAILY_COST_LIMIT_USD`, `LLM_MONTHLY_COST_LIMIT_USD`
- `FLOWER_USER`, `FLOWER_PASSWORD` (no defaults accepted)

Optional production settings:

- project/user LLM limit variables documented in `docs/llm-cost-control.md`
- `SENTRY_DSN`, `SENTRY_ENVIRONMENT`, and `SENTRY_TRACES_SAMPLE_RATE`
- `BACKUP_DIR` and `RETENTION_DAYS` for host-side PostgreSQL backups (default
  `./backups`, 7 days)
- `GRAFANA_ADMIN_PASSWORD`, `GRAFANA_ADMIN_USER` (monitoring overlay)
- `NEXT_PUBLIC_API_URL` (`/api` in production — never a secret)

Validate before deploying: `python scripts/maintenance/validate_production_config.py`.
Do not commit real secret values and never copy the development `.env` to
production.

## Kubernetes Secret (k8s)

The `k8s/` kustomization expects a Secret named `content-automation-secrets`
to exist in the `content-automation` namespace. A placeholder template is at
`k8s/secrets.example.yaml`; it is NOT part of the kustomization and is never
rendered or applied. kustomize renders successfully without the Secret present;
you must create it before `kubectl apply` so that the workloads start.

Required keys (referenced by `secretKeyRef`):
`postgres-password`, `secret-key`. Optional keys (mounted with
`optional: true`): `anthropic-api-key`, `openai-api-key`, `gemini-api-key`,
`redis-password`.

Create the Secret locally with placeholders replaced (use clipboard or a
file-mode secret; never echo literal values into a shell whose history is
shared):

```bash
kubectl create secret generic content-automation-secrets \
  --namespace content-automation \
  --from-literal=postgres-password='<generate-a-strong-password>' \
  --from-literal=secret-key='<generate-a-random-secret-key>' \
  --dry-run=client -o yaml > k8s/secrets.yaml

kubectl apply -f k8s/secrets.yaml
```

`k8s/secrets.yaml` is git-ignored; never commit it.

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
BACKUP_DIR=./backups RETENTION_DAYS=7 \
  scripts/maintenance/backup_database.sh
```

Backups are written on the Docker host, not inside an ephemeral container.
Copy them to encrypted off-server or object storage. Validate a backup without
changing the database:

```bash
scripts/maintenance/restore_database.sh backups/content_automation_TIMESTAMP.dump
```

Restore only during a maintenance window:

```bash
scripts/maintenance/restore_database.sh \
  backups/content_automation_TIMESTAMP.dump --confirm
docker compose -f docker-compose.prod.yml --profile migrate run --rm migrate
docker compose -f docker-compose.prod.yml up -d api worker celery-beat frontend nginx
docker compose -f docker-compose.prod.yml exec -T api \
  python scripts/maintenance/production_smoke_check.py \
    --api-url http://api:8000 \
    --frontend-url http://frontend:3001 \
    --nginx-url http://nginx
```

## Known Phase 2 Limitations

- Generated exports still use local storage.
- Sentry remains optional and must be configured by the operator.
- Backups require an external scheduler and off-server replication.
- Legacy plaintext WordPress credentials remain readable until updated or
  migrated under an operator-controlled credential rotation.
- Prompt-injection policy and moderation boundaries remain future work.
- Full browser E2E coverage is not yet required by CI.
