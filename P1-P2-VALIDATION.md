# Smarlux P1/P2 Validation Record

**Validation date:** 2026-08-01
**Baseline:** Smarlux P0 Production Candidate v1

## Freshly executed in the packaging environment

| Gate | Result |
|---|---|
| P1/P2 focused unit tests | PASS — 7 tests |
| P0 static release gate | PASS — 9 passed, 2 environment warnings |
| P1/P2 architectural invariants | PASS |
| Unified launch static gate | PASS — 8 passed, 1 environment warning |
| Python syntax compilation | PASS |
| Alembic migration graph | PASS — 10 revisions, single head `20260801_001` |
| TypeScript/TSX syntax transpilation | PASS — 40 files |
| Shared API client strict semantic TypeScript check | PASS |
| Docker Compose YAML parsing | PASS |
| Prometheus alert YAML parsing | PASS |
| Grafana dashboard JSON parsing | PASS |
| Git whitespace validation | PASS |
| High-confidence secret-pattern scan | PASS |
| Shell script syntax | PASS |
| Browser canary Python compilation | PASS |
| PostgreSQL p95 ordered-set expression compilation | PASS |

## Focused P1/P2 test coverage

`tests/test_p1_p2_launch_quality.py` verifies:

1. deterministic, explainable, read-only SEO intelligence;
2. stale-data confidence degradation and insufficient-data classification;
3. unknown-project rejection;
4. critical prioritization of stale WordPress work;
5. bounded lookback and bounded failure retrieval;
6. JSON-safe shared snapshot persistence with TTL;
7. API continuity when the snapshot cache is unavailable;
8. warning classification for connected but never-successful Search Console sync;
9. project-scoped summaries cannot overwrite the global metrics snapshot;
10. fixed-cardinality Prometheus rendering;
11. malformed/missing snapshots fail closed through availability metrics;
12. unknown integration/state values cannot create arbitrary labels.

Seven test functions cover these related assertions.

## Environment constraints

The packaging container uses Python 3.13.5, while this project supports Python 3.11–3.12. It does not contain the complete locked runtime environment, including asyncpg, Celery, Redis client packages, pgvector, tenacity, and structlog.

The focused P1/P2 tests were executed with a minimal import-only Redis stub. The tested service logic uses a fake cache and performs no live Redis operation.

A fresh `npm ci` could not complete because the available internal package mirror did not provide `yocto-queue@0.1.0`. Therefore, the changed frontend was freshly syntax-transpiled, but a dependency-backed frontend typecheck, lint, and production build was not rerun in this packaging container.

The broader P0 dependency-backed suite was not represented as freshly rerun here because its import graph requires the missing locked environment. The inherited P0 validation record remains in the package; production promotion must use the unified full gate in a supported clean environment.

## Required environment-dependent gates

Run:

```bash
export RUN_BACKUP_RESTORE_GATE=1
export RUN_BROWSER_GATE=1
scripts/maintenance/launch_release_gate.sh --full
```

The full gate requires:

- Python 3.11 or 3.12;
- all locked Python and frontend dependencies;
- focused P0/P1/P2 tests;
- frontend typecheck, lint, and production build;
- Docker Compose interpolation;
- disposable PostgreSQL backup/restore drill;
- read-only browser canary against the deployed candidate.

Live release gates still required:

- WordPress staging draft/create/update/idempotency/reconciliation canary;
- Google OAuth consent/property/sync/revocation canary;
- real alert-route delivery test;
- controlled staging soak and canary promotion.

## Validation interpretation

The source and packaging checks support classification as a **launch-ready production candidate**. They do not prove zero faults in an untested deployment, external account, network, browser fleet, or credential configuration.
