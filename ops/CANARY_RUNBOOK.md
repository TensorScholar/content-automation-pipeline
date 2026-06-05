# Canary Runbook (Minimal, Production-Focused)

This runbook is intentionally short and operational.
Use it before every production release.

## 1) Pre-Canary Gate

Run the existing pre-launch gate first:

```bash
python3 scripts/maintenance/pre_launch_check.py
```

If it reports critical failures, stop rollout.

## 2) Start Canary (10%)

Deploy one new API pod/instance while keeping stable instances active.
Route about 10% of traffic to canary.

Required checks for 15-20 minutes:

- API health remains healthy.
- Redis and Postgres stay healthy.
- Celery workers are online.
- Task processing reaches terminal states (SUCCESS/FAILURE), no stuck backlog.
- No spike in HTTP 5xx.

## 3) Execute Canary Smoke

Use the smoke script against the canary API endpoint:

```bash
CANARY_API_URL=https://your-canary-api.example.com \
CANARY_EMAIL=manager@smarlux.com \
CANARY_PASSWORD='your-password' \
bash scripts/maintenance/canary_smoke_check.sh
```

The script validates:

- auth token flow
- project create/list
- async content task submit
- task status polling to terminal state

## 4) Promote in Two Steps

If canary is stable:

1. Increase traffic from 10% to 50% and watch for 15 minutes.
2. Increase traffic from 50% to 100%.

## 5) Rollback Rules (Immediate)

Rollback immediately if any is true:

- sustained API 5xx rate > 10% for 2+ minutes
- no healthy Celery workers
- Redis or Postgres unavailable
- repeated stuck tasks (PENDING/STARTED beyond normal processing window)
- auth failures spike due to token/session regressions

## 6) Post-Release Verification

Run:

```bash
python3 scripts/maintenance/pre_launch_check.py
```

Then manually confirm in UI:

- Tasks page live updates
- WordPress publish action (draft first)
- Monitoring panel dependency statuses
