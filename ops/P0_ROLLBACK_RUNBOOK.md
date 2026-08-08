# Smarlux P0 Rollback Runbook

This runbook prioritizes application rollback without unnecessary database
restoration. Migration `20260801_001` is additive and is designed so the prior
application can normally run while the added tables and columns remain present.

## Trigger conditions

Roll back for duplicate publication, unauthorized public publication, corrupted
integration state, unrecoverable queue behavior, migration/data-integrity
failure, widespread authentication failure, or a severity-1/2 critical UI defect.

## 1. Contain side effects

Before replacing containers:

- disable or remove user access to live WordPress publication;
- stop new Search Console manual sync requests;
- preserve API, worker, Celery, PostgreSQL, Redis, and nginx logs;
- record the failing commit, image digest, task IDs, article IDs, project IDs,
  publishing attempt IDs, and Search Console sync-run IDs.

Do not delete failed or retrying records. They are operational evidence and may
be needed for reconciliation.

## 2. Stop schedulers and workers first

```bash
docker compose -f docker-compose.prod.yml stop celery-beat worker
```

Stopping workers before the API prevents new integration jobs from executing
while containment is in progress.

## 3. Roll back application images

Set `IMAGE_TAG` to the recorded known-good immutable tag and start the previous
application version:

```bash
export IMAGE_TAG="<known-good-tag>"
docker compose -f docker-compose.prod.yml up -d \
  api worker celery-beat frontend nginx
```

Run smoke checks immediately. In the normal rollback path, leave migration
`20260801_001` in place because its schema additions are non-destructive and an
unnecessary downgrade introduces more risk.

## 4. Database downgrade decision

Do not run `alembic downgrade` automatically.

Downgrade only when all of the following are true:

- the failure is caused by the new schema itself;
- the rollback has an approved maintenance window;
- the pre-migration backup has passed the disposable restore drill;
- no production data written to the new integration tables must be retained;
- a reviewer has inspected the migration downgrade path.

If those conditions are not met, retain the additive schema and roll back only
the application images.

## 5. Restore decision

Restore the database only for proven destructive corruption or an incompatible
migration that cannot be repaired safely. A restore discards data written after
the backup timestamp and therefore requires an explicit recovery decision.

Validate the selected backup first:

```bash
scripts/maintenance/verify_backup_restore.sh \
  backups/<approved-backup>.dump \
  --confirm-disposable-restore
```

Then follow the controlled restore procedure in
`docs/production-deployment.md`. Record the recovery point and all discarded
post-backup events.

## 6. WordPress reconciliation after rollback

A remote post can exist even when the local request reported failure. Before any
manual re-publication:

- search by stored WordPress post ID;
- search by deterministic slug;
- inspect the publishing attempt and idempotency key;
- compare remote title, status, URL, and modified timestamp;
- repair local state or remove the staging post only after an operator decision.

Never retry by manually creating another post before duplicate reconciliation.

## 7. Search Console containment after rollback

- disconnect affected projects if token handling is suspect;
- preserve sync-run records;
- revoke the OAuth grant at Google when credential exposure is possible;
- rotate the OAuth client secret if it may have leaked;
- do not delete previously imported performance snapshots unless an audit proves
  they are invalid.

## 8. Recovery verification

After rollback, verify:

- API/frontend/nginx smoke checks;
- login and project access;
- WordPress live publication remains disabled or explicitly gated;
- no integration task remains unexpectedly active;
- worker and beat topology is correct;
- monitoring and alert routing are operational;
- the known-good release commit and image digest are recorded.

Open an incident record before attempting a corrected release.
