# Smarlux Launch Runbook

This runbook promotes one immutable P0+P1+P2 candidate using Docker Compose. Kubernetes and desktop packaging are intentionally outside this launch path.

## 1. Freeze evidence

Record:

```bash
git rev-parse HEAD
git status --short
git tag --points-at HEAD
docker image inspect <IMAGE> --format '{{index .RepoDigests 0}}'
```

The working tree must be clean. Use immutable image digests, not floating tags.

## 2. Run the full launch gate

From a clean checkout with Python 3.11/3.12 and locked dependencies:

```bash
export RUN_BACKUP_RESTORE_GATE=1
export RUN_BROWSER_GATE=1
export APP_URL="https://<staging-host>"
export CANARY_EMAIL="<read-only-or-manager-canary-user>"
export CANARY_PASSWORD="<secret-from-secure-store>"

scripts/maintenance/launch_release_gate.sh --full
```

Do not continue on any failure.

## 3. Pre-deployment controls

- Rotate credentials previously exposed to terminals, screenshots, files, or chat.
- Verify `SECRET_KEY` and `CREDENTIAL_ENCRYPTION_KEY` are production values.
- Confirm PostgreSQL and Redis credentials are not defaults.
- Confirm Google OAuth redirect/return URLs use the final HTTPS host.
- Confirm the WordPress account uses an Application Password and minimum necessary permissions.
- Confirm public publishing remains approval-gated.
- Confirm RedBeat has one effective scheduler lock.
- Confirm alert routes reach a real operator.

## 4. Backup and migration

1. Create an encrypted off-host database backup.
2. Record backup identifier and checksum.
3. Run the disposable restore verification.
4. Put mutating traffic into maintenance/read-only mode if required by the migration plan.
5. Run Alembic exactly once:

```bash
alembic upgrade head
python scripts/maintenance/validate_migration_graph.py
```

Expected head:

```text
20260801_001
```

## 5. Start services

Use the production Compose definition and immutable image reference:

```bash
docker compose -f docker-compose.prod.yml up -d postgres redis
docker compose -f docker-compose.prod.yml run --rm api alembic upgrade head
docker compose -f docker-compose.prod.yml up -d api worker celery-beat frontend nginx
```

Use actual service names from `docker compose ... config --services` if the deployment overlay changes them.

## 6. Infrastructure smoke

Require:

- API readiness returns 200;
- frontend returns 200 through nginx;
- PostgreSQL and Redis healthy;
- integrations worker active and consuming `integrations`;
- one Celery Beat/RedBeat scheduler effective;
- Prometheus scraping API/worker targets;
- `integration_snapshot_available` equals `1`;
- `integration_snapshot_age_seconds` remains below `900`;
- Grafana durable integration panels load without missing-series errors;
- no critical alert firing except an explicitly simulated test.

## 7. Read-only browser canary

Run:

```bash
python scripts/maintenance/browser_release_canary.py
```

Review screenshots and `report.json` in `artifacts/launch-canary/`. Any unclassified browser exception, failed API request, or horizontal overflow blocks promotion.

## 8. WordPress canary

Against a disposable staging WordPress site:

1. qualification/preflight;
2. dry-run;
3. draft create;
4. repeat same request and confirm no duplicate;
5. update the existing draft;
6. simulate temporary timeout/5xx;
7. verify retry or reconciliation recovers state;
8. verify remote post ID, URL, slug and status;
9. only then perform one explicitly approved public publication if production policy allows it.

Any silent success, duplicate post, unverified remote state, or approval bypass blocks launch.

## 9. Search Console canary

Using a canary Google account/property:

1. start OAuth;
2. verify exact read-only scope;
3. select an accessible property;
4. start one completed-window sync;
5. verify durable run status and imported snapshots;
6. repeat the same sync and confirm idempotency;
7. verify truncated/partial coverage is visible;
8. revoke/disconnect and verify safe failure state;
9. reconnect only if required.

No Google write scope is permitted.

## 10. Controlled production promotion

- Start with internal or allowlisted users.
- Keep public WordPress publishing explicit and approval-gated.
- Observe at least one complete operational window.
- Watch error rate, queue depth, stale items, durable failure rate, p95 latency, snapshot age, database connections, Redis memory, and LLM cost.
- Promote gradually only when no severity-1/2 issue is open.

## 11. Launch acceptance

Launch is accepted only when:

- full gate is green;
- backup restore is proven;
- browser canary is green;
- WordPress and Search Console canaries are green;
- alert routing is proven;
- rollback has been rehearsed;
- immutable commit and image digest are recorded;
- final production approval is recorded.
