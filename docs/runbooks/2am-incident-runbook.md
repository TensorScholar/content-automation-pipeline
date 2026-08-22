# 2 AM Incident Runbook — Smarlux Content OS (Single VPS, Docker Compose)

> **Keep this printed or in `docs/runbooks/` on the VPS.** You will open this at 2 AM with a firing alert, no IDE, and `ssh` only.

**Launch profile:** Stage 14 Recommended — 4 vCPU / 8 GB / 40 GB, `docker-compose.prod.yml` (2 api × 4 workers, 3 workers × 4) + optional `docker-compose.monitoring.yml`. Single Ubuntu VPS. External Postgres/Redis supported via `DATABASE_URL`/`REDIS_URL` overrides (Stage 13) — then `postgres`/`redis` containers are unused.

---

## 0. 60-Second Triage

Run one command immediately. It collects everything you need to decide recovery, without mutating state:

```bash
./scripts/maintenance/ops_snapshot.sh
# or bundle for handoff:
./scripts/maintenance/ops_snapshot.sh --out /tmp
```

**Read the output top-to-bottom:**

| Line in snapshot | What it tells you | Decision |
|---|---|---|
| `DOCKER PS` shows `Up (healthy)` for `api`, `postgres`, `redis` | Stack is up | Go to §1 (symptom table) |
| Any service `Restarting` or `Exited` | Crash loop | Go to §2 |
| `/health` returns `{"status":"healthy"}` | App layer healthy | Likely capacity or integration issue |
| `/health` returns `degraded` or curl fails | API or dependency down | Go to §1 |
| `pg_stat_activity count` ≥ 80 or `REDIS used_memory` > 400 MB | Resource saturation | Go to §4 |
| `PROMETHEUS alerts firing: N` | Which alerts | Go to §1 table |
| `DISK` shows `Use%` ≥ 85% | Disk pressure | Go to §4 |
| `RECENT LOGS` tail | Error burst | Keep for post-mortem |

If you cannot `ssh` at all, this is a **VPS/host** incident — escalate to host provider console, then return here.

---

## 1. Symptom → One-Liner → Recovery

| # | Alert / Symptom (Prometheus or `curl`) | One-liner to confirm | Recovery (copy-paste) | When to rollback instead |
|---|---|---|---|---|
| **A1** | `ServiceDown` / `curl http://127.0.0.1/health` → `000` / nginx `502` | `docker compose -f docker-compose.prod.yml ps` — which service is not `healthy`? | `docker compose -f docker-compose.prod.yml up -d --no-deps api postgres redis && sleep 30 && curl -fs http://127.0.0.1/health` | If API restarts 3× in 5 min → go to §3 Rollback |
| **A2** | `PostgresDown` / `pg_isready` fails | `docker logs --tail 50 content-automation-postgres-prod` | `docker compose -f docker-compose.prod.yml restart postgres && sleep 10 && docker compose -f docker-compose.prod.yml ps` — DB is `pgvector/pg16`, WAL is `appendonly`; data survives restart. Check `df -h` — if disk full, free logs first (§4). | Never `down -v` (destroys volume). If data corruption suspected, go to §5 Restore. |
| **A3** | `CeleryWorkerDown` / `celery inspect ping` → no `pong` | `docker compose logs --tail 50 worker` | `docker compose -f docker-compose.prod.yml restart worker && sleep 20 && docker compose exec -T worker celery -A orchestration.celery_app.app inspect ping --timeout=5` | If worker loops `Exited (137)` OOM → reduce load (§4) then restart |
| **A4** | `CeleryTaskBacklog >100` (Warn) / `>300` (Crit) | `docker compose exec -T worker celery -A orchestration.celery_app.app inspect active_queues --timeout=5` + check `celery_queue_length` in Prometheus | Do **not** purge queues blindly. Let workers drain. If backlog grew after a bad deploy → §3 Rollback. If LLM rate-limited → check `llm_cost_total` and wait. | |
| **A5** | `HighErrorRate` / `APIErrorCritical` (>10% 5xx 2 min) | `docker compose logs --tail 30 api \| grep -i "ERROR\|Traceback"` | If errors started after deploy → §3 Rollback immediately. Otherwise `docker compose restart api` and watch `curl -fs http://127.0.0.1/health` every 10 s. | |
| **A6** | `IntegrationSnapshotStale` / `IntegrationStaleWork` | `curl -fs http://127.0.0.1:8000/system/metrics` (auth required) → `integration_snapshot_age_seconds` | `docker compose restart celery-beat` — beat refreshes `integration_snapshot_age_seconds` every 5 min. If stale >15 min → check `docker logs celery-beat`. | |
| **A7** | `HostDiskSpaceLow 85%` / `Critical 95%` | `df -h` ; `docker system df` | `docker system prune --volumes=false -f` is **not** safe — use: `docker compose logs --help` not needed; truncate json logs: `truncate -s 0 $(docker inspect --format='{{.LogPath}}' content-automation-postgres-prod 2>/dev/null)` per service, or `journalctl --vacuum-size=200M` on host. Then §4. | |
| **A8** | `HighMemoryUsage >2GB` per container | `docker stats --no-stream` | Identify offender (`api` or `worker` with ML). `docker compose restart <service>` frees RSS (torch cache). If repeated, you are on wrong profile — see Stage 14 Profile 2 vs 3. | |

**Containment first (P0 runbook priority):**

1. Disable public WordPress publication at app/config layer before any replay.
2. Keep `postgres` and evidence (logs, `publishing_attempt_ids`, `sync_run_ids`) — never `docker compose down -v`.
3. Record `git rev-parse HEAD`, `docker image inspect <IMAGE> --format '{{index .RepoDigests 0}}'`, and `ops_snapshot.sh --out /tmp` bundle.

---

## 2. Host Reboot / VPS Power-Cycle

All app services are `restart: unless-stopped` — they auto-start after host reboot. `migrate` is `restart: no` + `profiles: ["migrate"]` — it never auto-runs (correct: migrations are explicit).

```bash
# After reboot:
docker compose -f docker-compose.prod.yml ps          # expect postgres+redis healthy first (~10s), then api/worker/beat (~60s, ML import), then frontend/nginx
curl -fs http://127.0.0.1/health                      # expect {"status":"healthy"} within 120s
./scripts/maintenance/ops_snapshot.sh                  # file for post-mortem
```

If `api` stays `health: starting` > 3 min: `docker logs content-automation-pipeline-api-1 --tail 50` — look for `Database connection failed` → check `POSTGRES_PASSWORD` / `DATABASE_URL` and `docker logs content-automation-postgres-prod`.

---

## 3. Rollback (Application)

Rollback is **image-only** by default. Migration `20260801_001` is additive — prior app runs with new schema.

```bash
# 1) Record current (failing) and target (known-good) digests
git rev-parse HEAD
docker image inspect content-automation-pipeline:<TAG> --format '{{index .RepoDigests 0}}'

# 2) Stop workers first (prevents new integration jobs during containment)
docker compose -f docker-compose.prod.yml stop celery-beat worker

# 3) Roll to known-good tag (immutable, not `latest`)
export IMAGE_TAG="<known-good-tag>"
docker compose -f docker-compose.prod.yml up -d api worker celery-beat frontend nginx

# 4) Smoke
docker compose -f docker-compose.prod.yml run --rm api python scripts/maintenance/production_smoke_check.py --api-url http://api:8000 --frontend-url http://frontend:3001 --nginx-url http://nginx
```

**Do NOT** `alembic downgrade` or `restore_database.sh --confirm` unless:
- new schema itself is the cause, AND
- pre-migration backup passed `verify_backup_restore.sh --confirm-disposable-restore`, AND
- maintenance window + reviewer approved (see `ops/P0_ROLLBACK_RUNBOOK.md`).

WordPress/Search Console external state must be reconciled by `remote_post_id` / `slug` / `idempotency_key` before any re-publication — never blindly replay.

---

## 4. Capacity & Resource Exhaustion

Stage 14 capacities: Profile 2 (4 vCPU/8 GB) is tight at 6.4–7.8 GB idle. Triggers to scale to Profile 3 (8 vCPU/16 GB, externalized DB):

- `pg_stat_activity count` ≥ 80, `pool_timeout 30s`, or `HighMemoryUsage` on `api`/`worker` repeated.
- Redis `used_memory >400 MB` or evictions >0 (`redis-cli info stats` → `evicted_keys`).
- Celery `queue_length >100` sustained, or p95 `llm_request_duration >30s`.
- Host `loadavg > vCPU` or `HostDiskSpaceLow`.

Immediate relief (no deploy):

```bash
# Free disk (logs + build cache)
docker system df
# json-file logs are 10m × 5 = 50 MB/service; truncate if disk ≥85%
truncate -s 0 $(docker inspect --format='{{.LogPath}}' content-automation-postgres-prod)
docker builder prune -f 2>&1 | tail -5

# Reduce memory pressure without code: scale in
docker compose -f docker-compose.prod.yml up -d --scale worker=1
# then restart to drop RSS
docker compose restart api worker
```

---

## 5. Backup & Restore (Data Safety)

*Backups are host-side `pg_dump --format=custom`, validated in a disposable DB — production data is never dropped by the drill.*

```bash
# Daily backup (cron on VPS, see docs/production-deployment.md):
BACKUP_DIR=./backups RETENTION_DAYS=7 ./scripts/maintenance/backup_database.sh
# → copy to encrypted off-site storage; record SHA-256

# Verify without mutating production:
./scripts/maintenance/verify_backup_restore.sh --confirm-disposable-restore
# → expect RESTORE_VERIFY_PASS

# Validate a specific file:
./scripts/maintenance/verify_backup_restore.sh backups/<file>.dump --confirm-disposable-restore

# Destructive restore — maintenance window, app stopped:
./scripts/maintenance/restore_database.sh backups/<file>.dump --confirm
docker compose -f docker-compose.prod.yml --profile migrate run --rm migrate
docker compose -f docker-compose.prod.yml up -d api worker celery-beat frontend nginx
```

If backup fails: check `df -h` (no space), `docker compose ps postgres` (not healthy), and `.env` `POSTGRES_PASSWORD` / `DATABASE_URL`.

---

## 6. Alerting Setup (Single VPS)

Prometheus scrapes `api:8000/metrics` + `postgres-exporter` + `redis-exporter` + `node-exporter`. `monitoring/alert_rules.yml` has 5 groups (error rate, Celery, LLM, cache, infra) but `prometheus.yml: alerting.alertmanagers.targets` is intentionally empty in the base overlay — **alerts fire but do not page** unless you wire one of:

1. **Grafana contact point (recommended for single VPS):** Grafana 11 → Alerting → Contact points → Slack/email/webhook. Import `alert_rules.yml` via `provisioning`.
2. **Alertmanager overlay:** add `prom/alertmanager` service and point `prometheus.yml` at it.
3. **Cron poller (zero-dependency):** add to VPS crontab:

```cron
*/2 * * * * /usr/bin/curl -fs http://127.0.0.1:9090/api/v1/alerts 2>/dev/null | /usr/bin/python3 -c "import json,os,sys,urllib.request; d=json.load(sys.stdin); f=[a for a in d.get('data',{}).get('alerts',[]) if a.get('state')=='firing']; print('\n'.join(a['labels']['alertname']+':'+a['labels'].get('severity','') for a in f))" | grep -q . && curl -fs -X POST -H 'Content-Type: application/json' -d "{\"text\":\"Smarlux alerts firing\"}" "$ALERT_WEBHOOK_URL" || true
```

Without one of the above, **you must poll `http://VPS:9090/alerts` and `http://VPS:3000` dashboards manually** — not safe for 2 AM.

---

## 7. Escalation

1. Capture `ops_snapshot.sh --out /tmp` bundle.
2. Note `git rev-parse HEAD`, image digest, firing alert, and last deploy time.
3. Page operator via the wired Grafana/webhook channel (§6).
4. If no paging is wired, you are the pager — keep `http://VPS:9090/alerts` open in a tab.

---

*This runbook is the 2 AM companion to `ops/LAUNCH_RUNBOOK.md`, `ops/LAUNCH_ROLLBACK_RUNBOOK.md`, `ops/P0_ROLLBACK_RUNBOOK.md`, and `docs/production-deployment.md`. It does not replace them.*
