# Smarlux P1/P2 Launch Quality Implementation Report

**Candidate date:** 2026-08-01
**Baseline:** Smarlux P0 Production Candidate v1
**Objective:** improve operator visibility, UI resilience, measurable SEO intelligence, and launch evidence without unsafe automation, high-cardinality telemetry, or material request-path overhead.

## Engineering position

No software package can truthfully guarantee zero failures in an unknown production environment. This candidate instead provides bounded work, deterministic behavior, explicit failure states, durable recovery signals, low-cardinality observability, and repeatable promotion gates. No known P1/P2 source defect remains in the checks executed for this package.

## P1 — UI and operational reliability

### Durable integration operations service

Added a privileged operational aggregation path for WordPress publishing and Google Search Console synchronization:

```text
GET /system/integrations/operations
```

The service:

- queries both repositories concurrently;
- clamps the database lookback to 1–168 hours before issuing queries;
- returns at most ten recent failures per integration;
- distinguishes healthy, idle, warning, degraded, and critical states;
- treats stale durable operations as critical;
- treats connected Search Console properties without a successful sync as unproven;
- returns deterministic, bounded operator recommendations;
- keeps cache failures isolated from the API response;
- writes only the global summary to the shared metrics snapshot.

Project-scoped dashboard requests never overwrite the global operations snapshot.

### Repository operational summaries

WordPress and Search Console repositories now provide bounded SQL aggregation for:

- active-state counts;
- stale lease/run counts;
- recent terminal success/failure totals;
- latest successful completion;
- p95 terminal-operation duration;
- Search Console connection states and truncated runs;
- bounded, redacted recent failure details.

### Shared observability snapshot

Process-local worker metrics were deliberately rejected because Celery worker processes and multiple API workers do not share in-memory Prometheus collectors reliably.

The final design is:

```text
PostgreSQL durable attempts/runs
→ one Celery Beat refresh every five minutes
→ bounded JSON-safe Redis snapshot with 15-minute TTL
→ API /metrics cache read
→ fixed-cardinality Prometheus exposition
```

Prometheus scrapes do not execute WordPress/Search Console aggregation queries. Redis or snapshot failure is surfaced as an explicit availability metric instead of breaking `/metrics`.

Metrics:

```text
integration_snapshot_available
integration_snapshot_age_seconds
integration_durable_active_items
integration_durable_stale_items
integration_durable_health
integration_durable_recent_total
integration_durable_recent_succeeded
integration_durable_recent_failed
integration_durable_failure_rate
integration_durable_p95_duration_seconds
integration_durable_recent_truncated
```

Only fixed allow-listed labels are emitted: integration, active state, and health state. Project, URL, article, task, error, and user identifiers are prohibited from metric labels.

### Alerts and Grafana

Added launch alerts for:

- unavailable operations snapshot;
- stale operations snapshot;
- stale durable integration work;
- critical integration health;
- elevated failure rate with a minimum sample count.

Grafana includes:

- critical integration count;
- stale durable work;
- successful/failed outcomes over the durable 24-hour window;
- p95 operation latency over the durable 24-hour window.

### Monitoring UI

Added an Integration Reliability board to the existing Monitoring panel:

- independent WordPress and Search Console cards;
- active/stale counts and terminal success rate;
- localized health reasons and operator actions;
- bounded expandable failure details;
- EN, FA, and AR copy;
- RTL-compatible layout;
- accessible live status region;
- independent endpoint failure handling through `Promise.allSettled`.

A failed integration-operations request no longer blanks the rest of Monitoring.

### Frontend API reliability

The shared API client now provides:

- retries for GET only;
- maximum three retries;
- retries only for network faults and selected transient status codes;
- bounded exponential delay and capped `Retry-After` handling;
- one total request timeout and cancellation propagation;
- short-lived in-flight GET de-duplication;
- token-aware de-duplication without placing the raw token in the key;
- stable body/query/header serialization;
- `X-Request-ID` correlation;
- malformed JSON classification;
- no retry for POST, PUT, PATCH, or DELETE.

This improves transient read recovery without risking duplicate mutations.

## P2 — bounded, explainable product intelligence

### Deterministic SEO intelligence engine

Added a read-only portfolio intelligence endpoint:

```text
GET /projects/{project_id}/seo-intelligence
```

The engine combines existing first-party performance snapshots, Search Console state, sync history, article mapping, and improvement opportunities to provide:

- portfolio health score;
- measured article coverage;
- comparable-period trends;
- ranked opportunity queue;
- priority score and confidence;
- explicit score factors;
- estimated impact and effort;
- ordered, reversible action plans;
- freshness degradation;
- data-quality, mapping, truncation, and sync warnings.

Independent repository reads run concurrently.

### Intelligence safety contract

```text
uses_llm = false
performs_network_requests = false
rewrites_content = false
publishes_content = false
explanation_available = true
```

The engine recommends; it does not auto-edit or auto-publish. This preserves human approval, attribution, and rollback safety.

### Overhead controls

- maximum 2,000 snapshots per calculation;
- maximum 250 open opportunities loaded;
- maximum 100 ranked opportunities returned;
- maximum ten recent Search Console runs inspected;
- deterministic CPU-only scoring;
- no LLM call;
- no external HTTP request;
- no new runtime dependency;
- no P1/P2 database migration;
- no background polling in API workers.

### Project UI

Added an SEO Intelligence section to the existing Performance tab:

- health score;
- measured coverage;
- high-priority count;
- top recommended actions;
- confidence and data quality;
- localized warnings;
- EN, FA, and AR copy;
- mobile and RTL-compatible layout.

It refreshes after Search Console sync, manual import, opportunity dismissal, and explicit performance refresh.

## Launch engineering

Added:

- unified static/full release gate;
- read-only Playwright browser canary;
- HTTP error, network failure, browser exception, overflow, and RTL checks;
- immutable release runbook;
- conservative rollback runbook;
- external WordPress and Google canary requirements.

The browser canary contains no create, edit, publish, connect, disconnect, sync, dismiss, or import action.

## Explicitly deferred

Excluded to prevent launch risk or unnecessary overhead:

- autonomous content rewriting;
- unattended public publication;
- Search Console write scopes;
- query-level SEO warehouse;
- rank-tracking crawler;
- URL Inspection and sitemap administration;
- Kubernetes migration;
- Tauri changes;
- broad visual redesign;
- new LLM agents in request paths.

## Files added

```text
infrastructure/integration_metrics.py
services/integration_operations_service.py
services/seo_intelligence_service.py
tests/test_p1_p2_launch_quality.py
scripts/maintenance/p1_p2_static_invariants.py
scripts/maintenance/launch_release_gate.sh
scripts/maintenance/browser_release_canary.py
P1-P2-IMPLEMENTATION-REPORT.md
P1-P2-VALIDATION.md
ops/LAUNCH_RUNBOOK.md
ops/LAUNCH_ROLLBACK_RUNBOOK.md
```

## Release interpretation

This source is a **launch-ready production candidate**, not fabricated proof of an environment it has not seen. Production promotion requires the package’s full gate, external integration canaries, browser canary, backup/restore drill, alert routing verification, and controlled rollout to pass for one immutable commit and image digest.
