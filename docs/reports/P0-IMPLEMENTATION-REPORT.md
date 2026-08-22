# Smarlux P0 Implementation Report

## Scope

This production-candidate change set focuses only on P0 reliability:

- asynchronous and durable WordPress publication;
- duplicate prevention, remote verification, and reconciliation;
- read-only Google Search Console OAuth and data synchronization;
- critical frontend states for both integrations;
- migration, release, secret, and backup/restore gates.

It intentionally avoids another visual redesign, Kubernetes migration, Tauri
packaging, Search Console write features, URL Inspection, sitemap management,
rank tracking, and automated content rewriting.

## WordPress reliability

Publication requests now create or reuse a durable publishing attempt and enqueue
execution to the Celery `integrations` queue. The API returns an accepted state
while the frontend polls the durable attempt status.

The implementation adds:

- content-derived idempotency keys;
- one successful attempt per idempotency key at the database level;
- serialization of active article-side effects;
- leases for queued/running/retrying attempts;
- stale-attempt reconciliation;
- task-ownership tokens that suppress stale worker completion after reconciliation;
- protection against late retries or failures regressing a completed article;
- deterministic slug lookup and fail-closed duplicate search;
- authenticated capability qualification;
- draft/public permission checks;
- read-after-write verification of remote ID, slug, status, and URL;
- bounded retry classification;
- production target validation, DNS resolution rejection for non-public networks, and no redirect following;
- best-effort optional tags/SEO/schema metadata with structured warnings;
- manager authorization for scheduled/public requests.

The core post is created or updated first and verified before optional metadata is
attempted. Optional plugin incompatibility therefore cannot silently invalidate a
verified core post, and the result does not claim optional data was stored when it
was not.

## Google Search Console

The Search Console integration is read-only by construction. It adds:

- server-side OAuth authorization-code flow;
- exact granted-scope validation;
- one-time hashed OAuth state;
- encrypted refresh-token storage;
- accessible-property discovery and selection;
- manual and scheduled page-level Search Analytics synchronization;
- durable/idempotent sync runs with task ownership and reconciliation;
- bounded pagination with explicit truncation state;
- quota, permission, invalid-response, timeout, and revoked-token classification;
- disconnect/revoke behavior that removes the encrypted token while retaining
  audit history;
- preservation of manual CSV import as an independent fallback source.

The service rejects broader OAuth grants rather than accepting incremental or
write-capable access.

## Data and migration

Alembic revision `20260801_001` adds publication lease/status/verification fields,
strengthens successful-attempt uniqueness, and creates Search Console OAuth,
connection, property, and sync-run tables.

The migration is additive. It normalizes legacy publishing state and supersedes
historical duplicate-success records before creating the unique partial index.

## Frontend

The existing product design and navigation are retained. The P0 UI adds:

- WordPress queued, retrying, failed, and completed publication feedback;
- polling of durable publish status;
- Search Console connection/disconnection;
- property refresh and selection;
- manual sync and sync history;
- read-only labeling, truncation, quota, revoked, and error states;
- English, Persian, and Arabic copy with RTL support.

## Operations

New operational controls include:

- `scripts/maintenance/p0_release_gate.sh`;
- `scripts/maintenance/p0_static_invariants.py`;
- `scripts/maintenance/validate_migration_graph.py`;
- `scripts/maintenance/verify_backup_restore.sh`;
- deployment, rollback, and live-canary runbooks under `ops/`.

The worker topology explicitly consumes the `integrations` queue, reuses the worker process event loop for database-bound async tasks, and Celery Beat schedules WordPress and Search Console reconciliation.

## Production classification

This source is a P0 production candidate, not self-approving production evidence.
A clean Python 3.11/3.12 environment must run the full release gate, and real
WordPress, Google OAuth, PostgreSQL restore, browser, and controlled canary checks
must pass before public promotion.
