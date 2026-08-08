# Smarlux Launch Rollback Runbook

## Immediate rollback triggers

Rollback or disable mutating capabilities immediately on:

- duplicate or unverified WordPress publication;
- approval-gate bypass;
- migration corruption or data loss;
- repeated worker-loss loops;
- sustained critical integration health;
- authentication or authorization regression;
- unbounded error/latency increase;
- severity-1 or severity-2 incident.

## Containment order

1. Disable public WordPress publication at the application/configuration layer.
2. Stop new integration jobs while preserving PostgreSQL and evidence.
3. Keep read-only API access if safe.
4. Capture logs, metrics, task IDs, publishing attempt IDs, sync run IDs, commit SHA and image digest.
5. Do not delete durable attempts/runs to make the dashboard appear healthy.

## Application rollback

Deploy the last approved immutable image digest:

```bash
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d --no-deps api worker frontend nginx
```

Use the actual pinned previous digest in deployment configuration before this command.

## Migration decision

Do not automatically downgrade a production database. The P1/P2 layer adds no migration beyond the inherited P0 head `20260801_001`, but rollback must still respect the migration state of the deployed baseline.

If database restoration is required:

1. stop mutating services;
2. preserve the failed database and logs;
3. restore into a new database first;
4. verify schema, row counts, publication attempts and sync runs;
5. switch traffic only after validation;
6. retain the original database for investigation.

## External-state reconciliation

### WordPress

Before retrying publication after rollback:

- query by stored remote post ID;
- query deterministic slug/idempotency metadata;
- compare local and remote status;
- repair local state or mark for manual review;
- never blindly replay a public publication.

### Search Console

- preserve completed imported snapshots;
- mark stranded runs for reconciliation;
- do not duplicate completed project/property/window imports;
- reconnect only after the root cause is known.

## Rollback validation

Require:

- API/frontend readiness green;
- integrations queue stable;
- no new duplicate WordPress posts;
- no active stale task growth;
- authentication and role restrictions valid;
- browser read-only canary green;
- incident timeline recorded.
