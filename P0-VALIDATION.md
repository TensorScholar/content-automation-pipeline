# Smarlux P0 Validation Record

## Source-level validation executed in the build environment

The following checks are required before packaging and are rerun on the final
source tree:

- Git diff whitespace validation;
- Python compilation across application, migration, and test modules;
- single Alembic head verification (`20260801_001`);
- Compose YAML parsing;
- Celery `integrations` queue declaration and consumption;
- strict read-only Search Console architectural invariants;
- WordPress persistence, verification, and reconciliation hooks;
- frontend TypeScript/TSX syntax transpilation when full dependencies are absent;
- focused P0 service and distribution tests;
- high-confidence secret-pattern scan;
- shell-script syntax checks;
- final ZIP, patch-apply, rollback, and checksum verification.

## Focused automated test coverage

The P0-focused suite covers:

- one-time OAuth state and replay rejection;
- exact read-only scope enforcement and broader-scope rejection;
- production HTTPS OAuth URL requirements;
- Search Console pagination, truncation, duplicate delivery, invalid response,
  revoked credentials, missing projects, corrupted-token disconnect, and stale-task ownership rejection;
- WordPress duplicate lookup fail-closed behavior;
- read-after-write mismatch detection;
- content-revision idempotency;
- private/embedded-credential production target rejection;
- permission qualification;
- optional metadata degradation without core-post corruption;
- publication safety gates and performance-feedback bulk idempotency.

## Environment limitations

The packaging environment does not contain a complete project dependency install,
Docker, live PostgreSQL/Redis, WordPress credentials, or Google OAuth credentials.
It uses Python 3.13, while the project supports Python 3.11 and 3.12. Therefore the
following checks cannot be represented as passed by this source-only build:

- the complete dependency-backed backend test suite;
- a full Next.js dependency-backed typecheck, lint, and production build after
  the P0 UI changes;
- Docker Compose interpolation and service startup with real secrets;
- live migration and disposable restore;
- real WordPress draft/idempotency/failure-recovery canary;
- real Search Console OAuth/property/sync/revocation canary;
- final browser E2E and accessibility canary.

These are mandatory production gates, not optional future work. The package
includes `scripts/maintenance/p0_release_gate.sh --full` and the exact live-canary
runbook required to execute them in the deployment environment.

## Recorded source-candidate results

Final source validation in the packaging environment recorded:

```text
Focused P0 tests:                 63 passed
Python compileall:                PASS
Alembic revisions:                10
Expected/actual Alembic head:     20260801_001
Compose YAML parsing:             PASS
P0 architectural invariants:      PASS
Frontend TS/TSX syntax files:     40 transpiled
Static release gate:              9 passed, 2 warnings, 0 failed
Secret-pattern scan:              PASS
Git diff whitespace check:        PASS
Shell syntax checks:              PASS
```

The two static-gate warnings are expected and explicit: the packaging runtime is
Python 3.13 rather than the supported 3.11/3.12, and the local frontend dependency
installation is incomplete, so the full dependency-backed frontend build must be
rerun through `--full` in the deployment environment.
