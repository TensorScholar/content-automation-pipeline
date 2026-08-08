# Release Status

- **Current classification:** Staging only
- **Production approval:** Not granted
- **Last reviewed:** 2026-07-19

This repository contains production-oriented components, deployment definitions,
reliability controls, and test coverage. Their presence does not constitute a
production release decision. The current build may be used for local development
and controlled staging validation only.

Do not use the current build for:

- public general-availability traffic;
- unattended public publishing;
- production data without an approved backup and recovery procedure; or
- a production deployment built from an unreviewed or dirty working tree.

## Production Approval Gates

Production approval requires recorded evidence that all of the following gates
have passed for one immutable release commit and image digest:

1. The release scope is reviewed and the working tree is clean.
2. Backend, frontend, security, contract, integration, and browser end-to-end CI
   gates pass from a clean checkout.
3. Production credentials are stored securely and any exposed credentials have
   been rotated.
4. Database backups are automated, stored off-site, and proven by a restore
   drill against the approved RPO and RTO.
5. Production error tracking, metrics, logs, alert routing, and operational
   runbooks are active.
6. The selected LLM provider passes multilingual reliability, latency, cost, and
   failure-mode qualification.
7. The versioned article-quality and SEO benchmark passes deterministic,
   holdout, and human-review gates.
8. Critical FA, AR, and EN workflows pass browser, accessibility, responsive,
   RTL/LTR, and visual acceptance checks.
9. Migration, canary promotion, rollback, and post-release smoke procedures are
   exercised successfully.
10. No unresolved severity-1 or severity-2 issue remains, and the CTO records
    the final production decision.

Until every gate is satisfied, documentation and user communication must refer
to the system as a staging candidate, not as production ready.

## P0 Production Candidate Addendum — 2026-08-01

The P0 candidate adds durable WordPress publication, read-only Search Console
synchronization, focused reliability tests, release gates, and deployment/
rollback/canary runbooks. This improves release readiness but does not supersede
the approval gates above. Public promotion still requires a clean Python
3.11/3.12 full gate, real external integration canaries, a proven disposable
backup restore, browser acceptance, and an explicit production decision for the
immutable release commit and image digest.

## P1/P2 Launch Candidate Addendum — 2026-08-01

The P1/P2 candidate adds bounded integration operations intelligence,
low-cardinality metrics and alerts, partial-failure-safe Monitoring UI,
client-side GET retry/de-duplication/correlation, and deterministic first-party
SEO prioritization. The SEO engine performs no LLM call, external request,
content rewrite, or publication.

Operational aggregates are refreshed centrally by Celery Beat, persisted as a
bounded JSON-safe Redis snapshot, and rendered by the API metrics endpoint with
fixed allow-listed labels. Prometheus scrapes never trigger integration database
aggregation, and process-local worker counters are not used for cross-process
health. P1/P2 adds no new runtime dependency and no database migration beyond
the P0 head.

The source is classified as a launch-ready production candidate only after the
unified full release gate, real WordPress and Google canaries, browser
acceptance, backup/restore proof, alert routing, and controlled staging rollout
pass for one immutable commit and image digest. This addendum does not replace
the production approval gates above.
