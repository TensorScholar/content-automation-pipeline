# Smarlux P0 Deployment Runbook

This runbook deploys the P0 production candidate: fault-tolerant WordPress
publication, read-only Google Search Console synchronization, critical UI states,
and the release/backup gates required for a controlled launch.

The release is not approved by the presence of these files alone. Production
promotion requires the live canary evidence in `ops/P0_LIVE_CANARY.md` for one
immutable commit and image tag.

## 1. Preconditions

Use a clean checkout of the approved release commit. Record:

- Git commit SHA;
- image tag or digest;
- database backup path and checksum;
- operator and deployment timestamp;
- WordPress staging site used for the canary;
- Google Search Console property used for the read-only canary.

Production must use Python 3.11 or 3.12. Do not deploy from an uncommitted working
tree.

Required secrets and endpoints:

- `SECRET_KEY`;
- `CREDENTIAL_ENCRYPTION_KEY`, generated as a valid Fernet key;
- PostgreSQL, Redis, Celery broker, and Celery result credentials;
- selected LLM-provider credentials;
- WordPress credentials stored through the encrypted project configuration;
- all four Search Console OAuth variables when Search Console is enabled:
  `GOOGLE_SEARCH_CONSOLE_CLIENT_ID`,
  `GOOGLE_SEARCH_CONSOLE_CLIENT_SECRET`,
  `GOOGLE_SEARCH_CONSOLE_REDIRECT_URI`, and
  `GOOGLE_SEARCH_CONSOLE_FRONTEND_RETURN_URL`.

The Search Console redirect and frontend-return URLs must be HTTPS in production.
The Google OAuth client must register the redirect URI exactly.

## 2. Pre-deployment release gate

Run in the final deployment environment, with production-shaped dependencies and
configuration:

```bash
scripts/maintenance/p0_release_gate.sh --full
```

Stop if the command does not end with:

```text
P0_RELEASE_GATE_PASS
```

This gate verifies the migration graph, architecture invariants, Python runtime,
frontend typecheck/lint/build, focused P0 tests, Compose interpolation, encryption
key validity, required secrets, and high-confidence secret patterns.

## 3. Build immutable images

Use an immutable release tag rather than `latest`:

```bash
export IMAGE_TAG="p0-$(git rev-parse --short=12 HEAD)"
docker compose -f docker-compose.prod.yml build
```

Record the resulting image IDs or registry digests before promotion.

## 4. Backup before migration

Create the normal host-side PostgreSQL backup:

```bash
BACKUP_DIR=./backups RETENTION_DAYS=14 \
  scripts/maintenance/backup_database.sh
```

Copy the backup to encrypted off-host storage and record its SHA-256 checksum.
Then prove the backup is restorable in a disposable database:

```bash
scripts/maintenance/verify_backup_restore.sh \
  backups/<approved-backup>.dump \
  --confirm-disposable-restore
```

Stop if the command does not report `DISPOSABLE_RESTORE_PASS`.

## 5. Start data services and migrate explicitly

```bash
docker compose -f docker-compose.prod.yml up -d postgres redis

docker compose -f docker-compose.prod.yml \
  --profile migrate run --rm migrate
```

The expected Alembic head is:

```text
20260801_001
```

Do not allow replicated API containers to race migration execution.

## 6. Start application services

The worker must consume the `integrations` queue because WordPress publication
and Search Console synchronization run there.

```bash
docker compose -f docker-compose.prod.yml up -d \
  api worker celery-beat frontend nginx
```

Only one `celery-beat` instance may run.

Inspect service state and recent logs:

```bash
docker compose -f docker-compose.prod.yml ps

docker compose -f docker-compose.prod.yml logs \
  --tail=200 api worker celery-beat frontend nginx
```

## 7. Smoke checks

```bash
docker compose -f docker-compose.prod.yml run --rm api \
  python scripts/maintenance/production_smoke_check.py \
    --api-url http://api:8000 \
    --frontend-url http://frontend:3001 \
    --nginx-url http://nginx
```

Confirm API health, database readiness, Redis connectivity, worker availability,
frontend reachability, and no migration errors.

## 8. Controlled P0 canary

Execute every step in `ops/P0_LIVE_CANARY.md`. Keep public WordPress publication
disabled until the draft, idempotency, recovery, Search Console, and restore
checks pass.

The first WordPress operation must be a draft on a disposable staging site. The
first Search Console operation must use the read-only integration and a property
with non-sensitive test access.

## 9. Promotion

Promote gradually:

1. internal operator only;
2. one project with WordPress draft publication;
3. one Search Console property and one completed reporting window;
4. one manager-approved public WordPress article;
5. limited users;
6. broader traffic after the observation window.

During promotion, watch:

- integration queue depth and task age;
- retry and reconciliation counts;
- publishing-attempt status distribution;
- Search Console sync status, truncation, quota, and revoked-token errors;
- API 5xx and latency;
- PostgreSQL connections and locks;
- Redis availability;
- browser console and network errors;
- duplicate WordPress post reports.

## 10. Stop conditions

Pause or roll back immediately for any of these conditions:

- duplicate WordPress post creation;
- a public post without explicit manager authorization;
- local success when remote read-after-write verification failed;
- repeated stuck `running` or `retrying` attempts beyond their lease;
- Search Console requesting or receiving a scope other than
  `webmasters.readonly`;
- refresh-token decryption or revocation errors affecting multiple projects;
- migration, data-integrity, backup, restore, or authentication failure;
- severity-1 or severity-2 UI defect in a critical workflow.

Use `ops/P0_ROLLBACK_RUNBOOK.md` rather than improvising recovery commands.
