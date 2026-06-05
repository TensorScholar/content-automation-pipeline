# Reliability Hardening Log — 2026-04-09

This document records only the reliability work completed in this update.

## Backend Reliability Changes

- **SSE auth compatibility + security** (`api/routes/content.py`)
  - Added token query fallback for EventSource clients that cannot set `Authorization` headers.
  - Kept JWT validation in place via `decode_access_token`.
  - Enforced blacklist check for revoked tokens on SSE connections.
  - Added proper `WWW-Authenticate` response header for unauthorized access.
- **SSE resource safety** (`api/routes/content.py`)
  - Added disconnect detection (`request.is_disconnected()`) to stop SSE loop when client drops.

## Frontend Reliability Changes

- **Build blocker fix** (`frontend/src/components/app-shell.tsx`)
  - Removed broken trailing import and duplicate/unused imports causing TypeScript compile failure.
- **SSE contract + fallback** (`frontend/src/components/panels/tasks-panel.tsx`)
  - Switched streaming endpoint usage to `/content/task/{task_id}/events`.
  - Added robust message handling for SSE payloads.
  - Added polling fallback on SSE error with user-facing notice.
- **WordPress publish contract fix** (`frontend/src/components/panels/tasks-panel.tsx`)
  - Updated publish request to send required `project_id` and `post_status` query params.
  - Added guard when project context is missing.
- **Monitoring dependency parsing fix** (`frontend/src/components/panels/monitoring-panel.tsx`)
  - Added support for dependency statuses returned as plain strings from backend.
- **Language path stabilization** (`frontend/src/components/panels/content-studio-panel.tsx`)
  - Aligned request payload field to backend alias (`additional_instructions`).
  - Kept backend language contract (`en`/`fa`) and added explicit Arabic output instruction when Arabic is selected.
  - Fixed batch payload to use supported backend fields.

## Operational Reliability Additions

- **Canary rollout runbook** (`ops/CANARY_RUNBOOK.md`)
  - Added minimal rollout sequence: preflight, 10% canary, 50%, 100%, rollback conditions.
- **Canary smoke script** (`scripts/maintenance/canary_smoke_check.sh`)
  - Added minimal production smoke checks for auth, project create, async task submit, and terminal-state polling.
- **Alert thresholds hardening** (`monitoring/alert_rules.yml`)
  - Added critical API 5xx alert (`>10%` for `2m`).
  - Added critical Celery backlog alert (`>300` for `3m`).
- **Pre-launch checker modernization** (`scripts/maintenance/pre_launch_check.py`)
  - Updated default UI URL to Next.js port (`3001`) via `UI_URL`.
  - Updated dashboard health check to verify frontend proxy path (`/api/system/health`).
  - Improved dependency health parsing for string-based statuses and added Celery worker status visibility.
  - Added SSE route presence check in endpoint validation.

## Validation Performed

- `npm run typecheck` (frontend) — passed.
- `python3 -m py_compile api/routes/content.py scripts/maintenance/pre_launch_check.py` — passed.
- `python3 -m pytest tests/property_based/test_api_schemas.py -q` — passed.
- `python3 -m pytest tests/test_semantic_sanitization.py -q` — passed.
- `bash -n scripts/maintenance/canary_smoke_check.sh` — passed.
