# Smarlux Content OS — Client Handoff Package

> **Transfer record.** Client receives the product tomorrow. Another engineer maintains it the day after. A reviewer audits it next week.

**System** Smarlux Content OS — Content Automation Pipeline
**Release** `main` @ `ae4de87` (tag `v1.1.0`) — immutable; `git rev-parse HEAD` on delivery day is the record
**Classification** Staging-validated production candidate; production approval per `docs/release-status.md` (10 gates) remains with the CTO until live canaries pass
**Date** 2026-08-22
**Prepared by** Engineering — Principal, SRE
**For** Client operator, maintaining engineer, technical reviewer

---

## What Is Being Handed Off

A Next.js frontend, FastAPI backend, PostgreSQL + pgvector, Redis, Celery workers, nginx, and Docker Compose deployment — with configuration contract, runtime sizing, operational runbooks, and backup/restore tooling validated through Stages 13–15.5.

This package does not deploy the system. It makes the existing system transferable.

---

## Delivery Inventory

### Code

| Area | Path | Notes |
|---|---|---|
| Frontend | `frontend/` — Next.js 15, App Router, React 18, Tailwind, `next-themes` | Routes: `/` (login → AppShell), `/design-system`. Sections inside AppShell: `dashboard`, `projects`, `studio`, `tasks`, `users`, `monitoring`. Components in `src/components/ui/` and `src/components/panels/`. |
| Backend | `api/`, `core/`, `services/`, `intelligence/`, `knowledge/`, `orchestration/`, `infrastructure/` | FastAPI (`api/main.py`), Pydantic models (`core/models.py`), 80+ endpoints across `auth`, `content`, `projects`, `search_console`, `system`. |
| Database | `alembic/versions` (head `20260801_001`, 10 revisions), 18 tables, pgvector |  |
| Build | `Dockerfile` (python:3.11-slim, CPU-only torch), `frontend/Dockerfile` (node:22-alpine), `entrypoint.sh` |  |
| Config | `config/settings.py` (Pydantic `BaseSettings`), `.env.production.example` (canonical template) |  |

### Deployment

| Artifact | Path | Purpose |
|---|---|---|
| Production compose | `docker-compose.prod.yml` | Base stack — local PostgreSQL/Redis by default, external via `DATABASE_URL`/`REDIS_URL` overrides (Stage 13). |
| TLS overlay | `docker-compose.prod.https.yml` | Adds nginx TLS + certbot; requires `SERVER_NAME` and `nginx/ssl/live/<SERVER_NAME>/` |
| Monitoring overlay | `docker/docker-compose.monitoring.yml` | Prometheus, Grafana, 3 exporters — optional |
| Reverse proxy | `nginx/nginx.conf` + `nginx.http.conf` + `Dockerfile`/`docker-entrypoint.sh` | `envsubst` only `${SERVER_NAME}`; cert paths `live/${SERVER_NAME}/…` |
| Volumes | `postgres_data`, `redis_data`, `certbot_webroot` |  |

### Operations

| Artifact | Path | Use |
|---|---|---|
| Configuration contract | `docs/production-configuration.md` | §1 model, §2 required, §3 optional, §4 ownership, §5 issuance, §6 external PG, §7 external Redis, §8 frontend boundary, §9 creation, §10 validation, §11 never copy dev `.env` |
| Activation package | `docs/OPERATOR_ACTIVATION_PACKAGE.md` | 7 gates, one VPS, one ordered run (copy-paste; stop on red) |
| Deployment order | `docs/production-deployment.md` | 6 steps, smoke, rollback, metrics |
| Incident runbook | `docs/runbooks/2am-incident-runbook.md` | 60-second triage with `ops_snapshot.sh`, 8-row symptom table, reboot/rollback/capacity/backup/paging |
| Snapshot | `scripts/maintenance/ops_snapshot.sh` | Read-only bundle: `ps`, `stats`, `/health`, `pg_stat_activity`, `redis info`, Celery, disk, Prometheus, logs |
| Backups | `scripts/maintenance/backup_database.sh`, `restore_database.sh`, `verify_backup_restore.sh` | Host-side `pg_dump --format=custom`, disposable restore `smarlux_restore_verify_*` |
| Validation | `scripts/maintenance/validate_production_config.py` (63 checks) + `validate_compose_config.sh` | Contract + rendered compose proof |
| Sizing | `qa/stage14-runtime-profile-report.md` | Measured idle + 3 profiles |
| Hardening evidence | `qa/stage15-operations-hardening-report.md`, `qa/stage15.5-activation-package.md` |  |

### CI/CD

* `.github/workflows/ci.yml` — lint, test (real PG+Redis), `docker-build-validate` + `validate_production_config.py`, build & push GHCR on `main`, pre-deploy validate
* `.github/workflows/security-scan.yml` + `phase1-verification.yml` — security, Phase 1 gate
* `scripts/maintenance/p0_release_gate.sh --full` — full release gate; `--static-only` includes `validate_production_config.py --static`

---

## Deployment Model

Single Ubuntu VPS, Docker Compose. First client profile is Stage 14 **Profile 2 — 4 vCPU / 8 GB RAM / 40 GB SSD** (`api` 2 × 4 workers + 1, `worker` 3 × 4 slots, optional monitoring adds ~800 MiB). Profile 1 (2/4/20) boots but has no headroom; Profile 3 (8/16/80 + externalized DB) is for 200–500 articles/day. Details in `qa/stage14-runtime-profile-report.md` §3.

All services are `restart: unless-stopped`; `migrate` is `restart: no` + `profiles: ["migrate"]` — migrations are explicit (`docker compose --profile migrate run --rm migrate`), never on API start.

---

## Activation (Operator)

Execute **exactly** `docs/OPERATOR_ACTIVATION_PACKAGE.md`: 7 gates from laptop pre-flight through VPS clone, `.env` from `.env.production.example` (`chmod 600`, never copy dev `.env`), `validate_production_config.py --static` then rendered, `postgres`/`redis` → `migrate` → `api`/`worker`/`celery-beat`/`frontend`/`nginx` → smoke (`production_smoke_check.py`) → arm ops (snapshot baseline, `backup_database.sh` + `verify --confirm-disposable-restore` + cron `0 3 * * *` / `0 4 * * *` + one paging path from `2am-incident-runbook.md` §6). Gate outputs are `PASS` strings — stop on first failure.

---

## Operations (Day 2)

* **Triage at 2 AM:** `./scripts/maintenance/ops_snapshot.sh` then `docs/runbooks/2am-incident-runbook.md` §0 table. 8 symptoms have one-liner confirm + copy-paste recovery and rollback guard. Host reboot order is 120 s to `healthy`; rollback is image-only (`IMAGE_TAG` pin + `production_smoke_check.py`) unless all 5 downgrade conditions hold.
* **Daily:** backup cron + off-site `rclone`/`scp` + `curl http://127.0.0.1:9090/api/v1/alerts` shows `firing: 0`.
* **Weekly:** `verify_backup_restore.sh --confirm-disposable-restore` (disposable DB, no production impact) + Grafana `LLM cost` / `queue depth` / `pg_stat_activity` vs Stage 14 triggers (≥80 conns, >400 MB Redis, >100 queue → scale to Profile 3 / externalize).
* **Monitoring:** Prometheus scrapes `api:8000/metrics` internally (nginx blocks `/metrics` publicly); 5 alert groups in `monitoring/alert_rules.yml` (417 lines) but `prometheus.yml: alertmanagers.targets: []` — wire one of Grafana contact point / alertmanager overlay / 2-min cron poller (`ALERT_WEBHOOK_URL`) per runbook §6.

---

## Engineering (Maintenance)

**Code map for the next engineer:** `api/main.py` (app factory, middleware, health), `api/routes/` (auth/content/projects/search_console/system), `services/` (domain), `intelligence/` (LLM/KB), `knowledge/` (repositories), `core/models.py` (domain models), `infrastructure/database.py` (asyncpg + pgvector) + `redis_client.py` + `celery_app.py` (3 queues + beat).

**CI:** push to `codex/**` or `main` runs `ci.yml` (lint + `validate_migration_graph.py` + `p0_static_invariants.py` + Bandit, then test with real PG/Redis services, then `docker-build-validate` config + `validate_production_config.py` + builds). Main-only then builds GHCR `sha`/`latest`/date tags and `validate` (compose + k8s YAML).

**Testing:** `tests/test_settings_parsing.py` + `test_phase2_security_cost.py` cover settings fail-closed; P0 suite `tests/test_p0_integration_reliability.py` etc. run in `p0_release_gate.sh --full`.

**Frontend:** `frontend/src/i18n/provider.tsx` + `src/lib/api.ts` (`NEXT_PUBLIC_API_URL=/api` → nginx `/api` proxy; `NEXT_PUBLIC_GRAFANA_URL` optional). `next.config.mjs` rewrites `/api/:path*` → `API_PROXY_TARGET`. `next-themes` class strategy. Typecheck/lint/build are `npm run typecheck` / `lint` / `build` in `frontend/`.

**Migrations:** single head `20260801_001`; `alembic/env.py` rewrites `?ssl=require` → `?sslmode=require` for the sync path. Never run `alembic downgrade` without all 5 rollback conditions.

---

## Configuration

Created from `.env.production.example` on the VPS only, `chmod 600`. Required: `ENVIRONMENT=production`, `DEBUG=false`, `SERVER_NAME`, `POSTGRES_PASSWORD`, `REDIS_PASSWORD`, `DATABASE_URL`/`REDIS_URL`/`CELERY_*` (`${VAR:-local-default}` — external via `?ssl=require`/`rediss://`), `SECRET_KEY` (≥32 chars), `CREDENTIAL_ENCRYPTION_KEY` (Fernet), `ALLOWED_HOSTS` (`SERVER_NAME,localhost,127.0.0.1`), `CORS_ORIGINS` (explicit HTTPS), `LLM_PROVIDER` + its key, `FLOWER_USER`/`FLOWER_PASSWORD`, `GRAFANA_ADMIN_PASSWORD`. Optional: Search Console 4-tuple, Sentry/OTEL, cost limits, `NEXT_PUBLIC_API_URL=/api`. Full contract + generation commands in `docs/production-configuration.md` §2–5.

`validate_production_config.py` proves the contract: local defaults render when `DATABASE_URL` unset and external DSNs override when set (Compose v5.1.2 nested interpolation), required-var `:?` guards, and no `admin123`/`admin:admin` defaults.

---

## Security Posture

Fail-closed on every required secret (`${VAR:?…}`), CORS wildcard rejection in production, `SECRET_KEY` weak-value and length checks, `CREDENTIAL_ENCRYPTION_KEY` Fernet validation, Bandit + `pip-audit`/`safety` + Gitleaks in CI, JWT + RBAC, XSS helpers. WordPress credentials are Fernet-encrypted (`CREDENTIAL_ENCRYPTION_KEY` rotation requires re-encryption). TLS via `nginx/ssl/live/<SERVER_NAME>/fullchain.pem` from `certbot certonly --webroot`. Secrets are never logged; validation scripts print only `PASS` strings.

---

## Evidence

Stages 13–15.5 are staging-validated on an immutable release; production approval per `docs/release-status.md` (10 gates) remains with the CTO until live canaries pass on the delivery SHA.

* `scripts/maintenance/p0_release_gate.sh --static-only` includes `validate_production_config.py --static`
* `scripts/maintenance/validate_production_config.py` — rendered contract (local + external DSNs, HTTPS `ALLOWED_HOSTS` healthcheck, monitoring exporter defaults)
* `docker compose config --quiet` for `prod` and `prod.https` with CI env (with real `CREDENTIAL_ENCRYPTION_KEY` Fernet) — PASS
* `qa/stage13-production-configuration-report.md` (A–K), `qa/stage14-runtime-profile-report.md` (3 profiles), `qa/stage15-operations-hardening-report.md` (2 AM), `qa/stage15.5-activation-package.md`
* `git diff --check` clean at Stage 13 close; no `artifacts/` touched

---

## Limitations

Staging only until CTO gates pass; live WordPress + Search Console canaries are pending by design (disposable staging site/property, `ops/P0_LIVE_CANARY.md`). `P0`/`launch`/`canary` runbooks already exist for that window. Generated exports use local storage; full browser E2E is not yet CI-required (`docs/production-deployment.md §Known Limitations`).

---

## Handover Checklist (Print and Tick)

- [ ] `git rev-parse HEAD` recorded on both laptop and VPS: `ae4de87` (delivery tag `v1.1.0`)
- [ ] `git status --short` clean on both
- [ ] `docs/OPERATOR_ACTIVATION_PACKAGE.md` Gates 2–6 green (`PASS` strings)
- [ ] `curl http://127.0.0.1/health` → `{"status":"healthy"}`, `docker compose ps` all `healthy`
- [ ] `ops_snapshot.sh --out /tmp` bundle archived + `DISPOSABLE_RESTORE_PASS`
- [ ] One paging path tested (Grafana / alertmanager / cron webhook)
- [ ] `docs/runbooks/2am-incident-runbook.md` printed or on VPS in `docs/runbooks/`

---

## Support Transition

This package **is** the support transition. The next engineer needs only this file plus the linked docs and the repo itself. Open issues and severity-1/2 tracking live in the issue tracker; the next release records a new `git rev-parse HEAD` and `docker image inspect ... RepoDigests`.

**Primary references (in order):**

1. `docs/OPERATOR_ACTIVATION_PACKAGE.md` — ordered activation
2. `docs/production-configuration.md` — env contract
3. `docs/production-deployment.md` — deploy & rollback
4. `docs/runbooks/2am-incident-runbook.md` — incident
5. `qa/stage14-runtime-profile-report.md` — sizing

*End of handoff package. Execution time on a fresh VPS: ~15 min to Gate 5, +10 min to arm operations. Do not use for public production traffic until `docs/release-status.md` gate 10 is recorded.*
