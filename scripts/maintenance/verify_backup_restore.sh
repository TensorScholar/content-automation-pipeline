#!/usr/bin/env bash
# Verify a PostgreSQL backup by restoring it into a disposable database only.
# Production data is never dropped, overwritten, or stopped by this script.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.prod.yml}"
POSTGRES_SERVICE="${POSTGRES_SERVICE:-postgres}"
POSTGRES_USER="${POSTGRES_USER:-content_user}"
POSTGRES_DB="${POSTGRES_DB:-content_automation}"
EXPECTED_ALEMBIC_HEAD="${EXPECTED_ALEMBIC_HEAD:-20260801_001}"
CONFIRM=false
BACKUP_FILE=""
TEMP_BACKUP=""
RESTORE_DB="smarlux_restore_verify_$(date -u +%Y%m%d%H%M%S)_$$"

usage() {
  cat <<'EOF'
Usage:
  scripts/maintenance/verify_backup_restore.sh [BACKUP_FILE]
  scripts/maintenance/verify_backup_restore.sh [BACKUP_FILE] --confirm-disposable-restore

Without --confirm-disposable-restore, this performs non-destructive prerequisites
and backup-format validation only. With confirmation, the backup is restored into
a uniquely named disposable database, verified, and dropped automatically.
If BACKUP_FILE is omitted in confirmed mode, a temporary backup of the current
production database is created and tested.
EOF
}

for arg in "$@"; do
  case "$arg" in
    --confirm-disposable-restore) CONFIRM=true ;;
    --help|-h) usage; exit 0 ;;
    --*) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
    *)
      if [[ -n "$BACKUP_FILE" ]]; then
        echo "Only one backup file may be supplied" >&2
        exit 2
      fi
      BACKUP_FILE="$arg"
      ;;
  esac
done

command -v docker >/dev/null 2>&1 || { echo "Docker is required" >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "Docker Compose is required" >&2; exit 1; }
docker compose -f "$COMPOSE_FILE" ps --status running "$POSTGRES_SERVICE" | grep -q "$POSTGRES_SERVICE" \
  || { echo "PostgreSQL Compose service is not running" >&2; exit 1; }

cleanup() {
  if [[ "$CONFIRM" == true ]]; then
    docker compose -f "$COMPOSE_FILE" exec -T "$POSTGRES_SERVICE" \
      dropdb --if-exists --force --username="$POSTGRES_USER" "$RESTORE_DB" >/dev/null 2>&1 || true
  fi
  if [[ -n "$TEMP_BACKUP" ]]; then
    rm -f "$TEMP_BACKUP"
  fi
}
trap cleanup EXIT INT TERM

if [[ -z "$BACKUP_FILE" ]]; then
  if [[ "$CONFIRM" != true ]]; then
    echo "PREREQUISITES_PASS"
    echo "Supply a backup file, or re-run with --confirm-disposable-restore to create and test a temporary backup."
    exit 0
  fi
  TEMP_BACKUP="$(mktemp "${TMPDIR:-/tmp}/smarlux-backup-XXXXXX.dump")"
  chmod 600 "$TEMP_BACKUP"
  docker compose -f "$COMPOSE_FILE" exec -T "$POSTGRES_SERVICE" \
    pg_dump --format=custom --no-owner --no-acl \
    --username="$POSTGRES_USER" --dbname="$POSTGRES_DB" > "$TEMP_BACKUP"
  BACKUP_FILE="$TEMP_BACKUP"
fi

[[ -r "$BACKUP_FILE" && -s "$BACKUP_FILE" ]] \
  || { echo "Backup file is missing, unreadable, or empty: $BACKUP_FILE" >&2; exit 1; }

docker compose -f "$COMPOSE_FILE" exec -T "$POSTGRES_SERVICE" \
  pg_restore --list < "$BACKUP_FILE" >/dev/null

echo "BACKUP_FORMAT_PASS"
if [[ "$CONFIRM" != true ]]; then
  echo "No restore was performed. Add --confirm-disposable-restore for the isolated restore drill."
  exit 0
fi

docker compose -f "$COMPOSE_FILE" exec -T "$POSTGRES_SERVICE" \
  createdb --username="$POSTGRES_USER" --template=template0 "$RESTORE_DB"

docker compose -f "$COMPOSE_FILE" exec -T "$POSTGRES_SERVICE" \
  pg_restore --exit-on-error --no-owner --no-acl \
  --username="$POSTGRES_USER" --dbname="$RESTORE_DB" < "$BACKUP_FILE"

required_tables=(
  alembic_version
  projects
  generated_articles
  publishing_attempts
  search_console_connections
  search_console_sync_runs
)
for table in "${required_tables[@]}"; do
  exists="$(docker compose -f "$COMPOSE_FILE" exec -T "$POSTGRES_SERVICE" \
    psql --username="$POSTGRES_USER" --dbname="$RESTORE_DB" --tuples-only --no-align \
    --command="SELECT to_regclass('public.${table}') IS NOT NULL;")"
  [[ "$exists" == "t" ]] || { echo "Restore verification failed: missing table $table" >&2; exit 1; }
done

head="$(docker compose -f "$COMPOSE_FILE" exec -T "$POSTGRES_SERVICE" \
  psql --username="$POSTGRES_USER" --dbname="$RESTORE_DB" --tuples-only --no-align \
  --command="SELECT version_num FROM alembic_version LIMIT 1;")"
[[ "$head" == "$EXPECTED_ALEMBIC_HEAD" ]] \
  || { echo "Restore verification failed: Alembic head '$head' != '$EXPECTED_ALEMBIC_HEAD'" >&2; exit 1; }

integrity="$(docker compose -f "$COMPOSE_FILE" exec -T "$POSTGRES_SERVICE" \
  psql --username="$POSTGRES_USER" --dbname="$RESTORE_DB" --tuples-only --no-align \
  --command="SELECT COUNT(*) >= 0 FROM projects;")"
[[ "$integrity" == "t" ]] || { echo "Restore verification query failed" >&2; exit 1; }

echo "DISPOSABLE_RESTORE_PASS database=$RESTORE_DB alembic_head=$head"
echo "The disposable database will now be dropped automatically."
