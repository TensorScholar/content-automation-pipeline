#!/usr/bin/env bash
# ops_snapshot.sh — 2 AM evidence bundle
# Collects a single-host triage snapshot without mutating state.
# Safe to run on production at any time. No secrets are printed.
# Usage:
#   ./scripts/maintenance/ops_snapshot.sh               # prints to stdout
#   ./scripts/maintenance/ops_snapshot.sh --out /tmp    # writes bundle to /tmp/ops-snapshot-*.tar.gz
set -euo pipefail

OUT_DIR=""
if [[ "${1:-}" == "--out" && -n "${2:-}" ]]; then OUT_DIR="$2"; fi
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: $0 [--out DIR]"
  echo "  Collects docker ps, stats, health, pg/redis, disk, logs, prometheus alerts."
  exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/ops-snapshot-$TS-XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

say() { printf "\n===== %s =====\n" "$*"; }

{
say "HOST $TS  $(hostname 2>/dev/null || echo unknown)  $(date -u)"
say "GIT"
(cd "$ROOT" && git rev-parse HEAD 2>/dev/null || echo "no git")
(cd "$ROOT" && git status --short 2>/dev/null | head -20 || true)

say "DOCKER PS"
docker compose -f "$ROOT/docker-compose.prod.yml" ps 2>&1 | head -40 || docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>&1 | head -40

say "DOCKER STATS --no-stream"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}" 2>&1 | head -30

say "HEALTH — curl /health via localhost and via nginx"
curl -fsS --max-time 5 http://127.0.0.1/health 2>&1 | head -20 || echo "curl /health failed"
curl -fsS --max-time 5 http://127.0.0.1/api/health 2>&1 | head -20 || echo "curl /api/health failed"
echo "nginx /health: $(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1/health 2>&1 || echo fail)"

say "POSTGRES — pg_isready + pg_stat_activity count + database size"
docker compose -f "$ROOT/docker-compose.prod.yml" exec -T postgres pg_isready -U content_user -d content_automation 2>&1 | head -5 || echo "pg_isready failed"
docker compose -f "$ROOT/docker-compose.prod.yml" exec -T postgres psql -U content_user -d content_automation -c "SELECT count(*) AS connections FROM pg_stat_activity;" 2>&1 | head -10 || echo "pg_stat_activity failed"
docker compose -f "$ROOT/docker-compose.prod.yml" exec -T postgres psql -U content_user -d content_automation -c "SELECT pg_size_pretty(pg_database_size('content_automation'));" 2>&1 | head -10 || true

say "REDIS — ping + info memory + info replication"
REDIS_PW="${REDIS_PASSWORD:-}"
if [[ -z "$REDIS_PW" && -f "$ROOT/.env" ]]; then REDIS_PW="$(grep -E '^REDIS_PASSWORD=' "$ROOT/.env" 2>/dev/null | cut -d= -f2- | tr -d '"'\'' ' || true)"; fi
if [[ -n "$REDIS_PW" ]]; then
  docker compose -f "$ROOT/docker-compose.prod.yml" exec -T redis redis-cli -a "$REDIS_PW" ping 2>&1 | head -5 || echo "redis ping failed"
  docker compose -f "$ROOT/docker-compose.prod.yml" exec -T redis redis-cli -a "$REDIS_PW" info memory 2>&1 | grep -E "used_memory_human|used_memory_peak_human|maxmemory_human|maxmemory_policy|connected_clients" | head -10 || true
else
  echo "REDIS_PASSWORD not set — skipping authenticated redis-cli checks"
fi

say "CELERY — inspect ping + active queues (if worker is up)"
docker compose -f "$ROOT/docker-compose.prod.yml" exec -T worker celery -A orchestration.celery_app.app inspect ping --timeout=5 2>&1 | head -20 || echo "celery ping failed or worker down"
docker compose -f "$ROOT/docker-compose.prod.yml" exec -T worker celery -A orchestration.celery_app.app inspect active_queues --timeout=5 2>&1 | head -30 || true

say "DISK + INODES"
df -h 2>&1 | head -20
df -i 2>&1 | head -20

say "PROMETHEUS — targets and firing alerts (if monitoring overlay is up)"
curl -fsS --max-time 5 http://127.0.0.1:9090/api/v1/targets 2>&1 | python3 -c "import json,sys; d=json.load(sys.stdin); print('\n'.join(f\"{t['labels'].get('job','?')} {t['health']}\" for t in d.get('data',{}).get('activeTargets',[])[:10]))" 2>&1 | head -20 || echo "prometheus targets unavailable (monitoring overlay not running)"
curl -fsS --max-time 5 http://127.0.0.1:9090/api/v1/alerts 2>&1 | python3 -c "import json,sys; d=json.load(sys.stdin); a=d.get('data',{}).get('alerts',[]); print(f'alerts firing: {len(a)}'); [print(f\"{x.get('labels',{}).get('alertname')} {x.get('state')} {x.get('labels',{}).get('severity','')}\") for x in a[:20]]" 2>&1 | head -30 || echo "prometheus alerts unavailable"

say "RECENT LOGS -- tail 30 per service"
for svc in api worker celery-beat frontend nginx postgres redis; do
  echo "--- $svc ---"
  docker compose -f "$ROOT/docker-compose.prod.yml" logs --tail 30 "$svc" 2>&1 | tail -30 || echo "(no logs for $svc)"
done

say "END $TS"
} 2>&1 | tee "$TMP/snapshot.txt"

if [[ -n "$OUT_DIR" ]]; then
  mkdir -p "$OUT_DIR"
  tar -czf "$OUT_DIR/ops-snapshot-$TS.tar.gz" -C "$TMP" snapshot.txt 2>&1 | head -5
  echo "Bundle: $OUT_DIR/ops-snapshot-$TS.tar.gz"
else
  cat "$TMP/snapshot.txt" >/dev/null
fi
