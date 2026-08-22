# Smarlux Content OS — Operator Activation Package (Stage 15.5)

> **One document, one VPS, one ordered run.** Copy-paste each gate. Stop on first red. Do not improvise.

**Audience:** Client operator with `ssh` + `sudo` on a fresh Ubuntu VPS and access to the production secrets (LLM key, domain, WordPress app-password if used).

**What this is:** The executable glue between Stage 13 (configuration contract), Stage 14 (sizing), and Stage 15 (incident hardening). It turns those artifacts into a single activation sequence.

**What this is NOT:** A deployment — this package validates and arms the system. The actual `docker compose up` is one step inside it.

**Provenance:** `docs/production-configuration.md` (contract) + `.env.production.example` (template) + `validate_production_config.py` (63 checks) + Stage 14 Profile 2 + `docs/runbooks/2am-incident-runbook.md` + `scripts/maintenance/ops_snapshot.sh` + `backup_database.sh` / `verify_backup_restore.sh` / `restore_database.sh`.

---

## Recommended First Environment (Stage 14 Profile 2 — do not downsize)

* **OS:** Ubuntu 22.04/24.04 LTS, fresh VPS, `ssh` key only
* **Size:** **4 vCPU / 8 GB RAM / 40 GB SSD** (minimum that holds `api 2×4 workers` + `worker 3×4` at ~6.4–7.8 GB idle). Profile 1 (2 vCPU/4 GB/20 GB) boots but has no headroom; Profile 3 (8 vCPU/16 GB/80 GB) is for 200–500 articles/day.
* **Software on VPS:** `docker engine 24+` + `docker compose v2`, `git`, `python3.11`, `curl`, `jq`
* **Network:** `SERVER_NAME` DNS `A` record → VPS IP, ports `80`/`443` open, `5432`/`6379` closed
* **On your laptop:** this repo cloned, production secrets in a vault (never in chat/screenshots)

---

## Pre-Flight (2 minutes, on your laptop)

```bash
git rev-parse HEAD                          # record — this is your immutable release
git status --short                          # must be empty
git tag --points-at HEAD                    # expect v1.1.0 or your release tag
cat .env.production.example | head -5       # confirm template exists
ls docs/runbooks/2am-incident-runbook.md scripts/maintenance/ops_snapshot.sh  # hardening assets present
```

---

## Gate 1 — Clone on the VPS

```bash
ssh <operator>@<VPS_IP>
sudo apt update && sudo apt install -y docker.io docker-compose-plugin git python3-pip
git clone <your-repo-url> /opt/smarlux && cd /opt/smarlux
git rev-parse HEAD   # must match laptop
```

> **Expected:** same SHA on both sides. **Stop if** SHA differs or working tree dirty.

---

## Gate 2 — Create Production `.env` (never copy dev `.env`)

```bash
cp .env.production.example .env
chmod 600 .env
nano .env   # fill every empty value — see docs/production-configuration.md §2
```

Required secrets (generate **on the VPS**):
```bash
python3 -c "import secrets; print(secrets.token_urlsafe(48))"   # SECRET_KEY
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"  # CREDENTIAL_ENCRYPTION_KEY
openssl rand -base64 24   # POSTGRES_PASSWORD / REDIS_PASSWORD / FLOWER_PASSWORD
```

Required non-secrets: `SERVER_NAME`, `ALLOWED_HOSTS=<SERVER_NAME>,localhost,127.0.0.1`, `CORS_ORIGINS=https://<SERVER_NAME>`, `LLM_PROVIDER` + its key.

```bash
python3 scripts/maintenance/validate_production_config.py --static
```

> **Expected:** `PRODUCTION_CONFIG_VALIDATION_PASS` (53 checks, 0 failed). **Stop if** any failed — fix `.env` per the error.

---

## Gate 3 — Render Check (no containers started)

```bash
POSTGRES_PASSWORD=dummy REDIS_PASSWORD=dummy SECRET_KEY=dummy CREDENTIAL_ENCRYPTION_KEY=dummy FLOWER_USER=dummy FLOWER_PASSWORD=dummy SERVER_NAME=example.com \
  python3 scripts/maintenance/validate_production_config.py
```

> **Expected:** `PRODUCTION_CONFIG_VALIDATION_PASS` (63 checks) + local defaults and external-DSN overrides proven. This also covers the nested `${VAR:-…}` portability from Stage 13.

---

## Gate 4 — Build & Start Data Layer

```bash
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d postgres redis
# wait ~10 s
docker compose -f docker-compose.prod.yml ps   # both healthy
docker compose -f docker-compose.prod.yml --profile migrate run --rm migrate
# expected: "Migrations completed successfully!  ... head 20260801_001"
```

---

## Gate 5 — Start Application & Smoke

```bash
docker compose -f docker-compose.prod.yml up -d api worker celery-beat frontend nginx
# wait 60–120 s (ML cold import, start_period 180 s is intentional)
curl -fs http://127.0.0.1/health
# expected: {"status":"healthy","dependencies":{"database":"healthy","redis":"healthy"}}

docker compose -f docker-compose.prod.yml run --rm api \
  python scripts/maintenance/production_smoke_check.py \
    --api-url http://api:8000 --frontend-url http://frontend:3001 --nginx-url http://nginx
# expected: SMOKE_PASS
```

Optional HTTPS (only after `nginx/ssl/live/<SERVER_NAME>/` certs exist):
```bash
docker compose -f docker-compose.prod.yml -f docker-compose.prod.https.yml up -d
```

---

## Gate 6 — Arm Operations (the 2 AM hardening)

This gate is **what Stage 15 added** — it is the difference between “it runs” and “it wakes someone at 2 AM”.

### 6a — Baseline snapshot (file for handover)

```bash
./scripts/maintenance/ops_snapshot.sh --out /tmp
# expected: Bundle: /tmp/ops-snapshot-*.tar.gz  (contains ps, stats, /health, pg_stat_activity, redis info, alerts, logs --tail)
```

Keep this bundle — it is the post-mortem baseline.

### 6b — Backup + disposable restore drill (proves RPO)

```bash
BACKUP_DIR=./backups RETENTION_DAYS=7 ./scripts/maintenance/backup_database.sh
ls -lh backups/   # one *.dump, chmod 600
./scripts/maintenance/verify_backup_restore.sh --confirm-disposable-restore
# expected: DISPOSABLE_RESTORE_PASS  (restores into smarlux_restore_verify_*, then drops — production untouched)
```

Now wire the daily automation (on the VPS):

```cron
0 3 * * * BACKUP_DIR=/var/backups/smarlux RETENTION_DAYS=7 /opt/smarlux/scripts/maintenance/backup_database.sh && rclone copy /var/backups/smarlux remote:smarlux-backups  # or scp/rsync to off-site
0 4 * * * /opt/smarlux/scripts/maintenance/verify_backup_restore.sh --confirm-disposable-restore >> /var/log/smarlux-verify.log 2>&1
```

### 6c — Wire exactly one paging path (otherwise alerts are silent)

`monitoring/alert_rules.yml` has 5 groups but `prometheus.yml: alertmanagers.targets: []` — Prometheus will evaluate but not page. Pick **one**:

* **(Recommended) Grafana contact point:** Grafana 11 → Alerting → Contact points → Slack/email/webhook → import `monitoring/alert_rules.yml` via provisioning. Test: fire `HighErrorRate` → message arrives.
* **Alertmanager overlay:** add `prom/alertmanager` service and point `prometheus.yml` at it.
* **Cron poller (zero-dependency):** `crontab -e` →
  ```cron
  */2 * * * * /usr/bin/curl -fs http://127.0.0.1:9090/api/v1/alerts 2>/dev/null | /usr/bin/python3 -c "import json,sys; d=json.load(sys.stdin); f=[a for a in d.get('data',{}).get('alerts',[]) if a.get('state')=='firing']; print('\n'.join(a['labels']['alertname']+':'+a['labels'].get('severity','') for a in f))" | grep -q . && curl -fs -X POST -H 'Content-Type: application/json' -d "{\"text\":\"Smarlux alerts firing\"}" "$ALERT_WEBHOOK_URL" || true
  ```

Without one of the above, keep `http://<VPS>:9090/alerts` open — **not safe for 2 AM**.

### 6d — Print the runbook

```bash
cat docs/runbooks/2am-incident-runbook.md   # or print to paper
```

The 2 AM runbook’s §0 is the 60-second triage (`ops_snapshot.sh` table), §1 is the 8-row symptom→one-liner→recovery table, §5 is backup/restore, §6 is alert wiring. It is the companion to `ops/LAUNCH_RUNBOOK.md`.

---

## Gate 7 — Go / No-Go

Check all gates green:

- [ ] Gate 2: `validate_production_config.py --static` PASS
- [ ] Gate 3: `validate_production_config.py` (rendered) PASS
- [ ] Gate 4: `migrate` head `20260801_001`
- [ ] Gate 5: `curl /health` healthy + `production_smoke_check` PASS
- [ ] Gate 6a: `ops_snapshot --out` bundle archived
- [ ] Gate 6b: `DISPOSABLE_RESTORE_PASS`
- [ ] Gate 6c: one paging path tested (message received)
- [ ] `git status --short` clean, `IMAGE_TAG` immutable recorded, `docs/runbooks/2am-incident-runbook.md` on the VPS

**Go:** hand the VPS `ssh` + Grafana + `ops_snapshot.sh --out` bundle to the operator. **No-Go:** stay on staging; see `docs/release-status.md` gate 10 and `ops/LAUNCH_ROLLBACK_RUNBOOK.md` (image-only rollback, additive migration).

---

## After Activation — Daily / Weekly

* **Daily (cron):** backup + off-site + `curl http://127.0.0.1:9090/api/v1/alerts` shows `firing: 0`.
* **Weekly:** `verify_backup_restore.sh --confirm-disposable-restore`, review Grafana `LLM cost` / `queue depth` / `pg_stat_activity` vs Stage 14 triggers (≥80 conns, >400 MB Redis, >100 queue → scale to Profile 3 / externalize DB).

---

## Quick Reference

| Asset | Path | Use |
|---|---|---|
| Config contract | `docs/production-configuration.md` | authoritative env list |
| Template | `.env.production.example` | copy to `.env` on VPS |
| Validator | `scripts/maintenance/validate_production_config.py` | gates 2–3 |
| Deployment order | `docs/production-deployment.md` | gates 4–5 |
| 2 AM runbook | `docs/runbooks/2am-incident-runbook.md` | incident triage |
| Snapshot | `scripts/maintenance/ops_snapshot.sh` | gate 6a, every incident |
| Backups | `backup_database.sh` / `verify_backup_restore.sh` / `restore_database.sh` | gate 6b |
| Sizing | `qa/stage14-runtime-profile-report.md` | Profile 2 is this package |

*End of activation package. Execution time on a fresh VPS: ~15 minutes to Gate 5, +10 minutes to arm operations.*
