#!/usr/bin/env bash
#
# Create a PostgreSQL custom-format backup on the Docker host.
#
# Usage:
#   BACKUP_DIR=./backups RETENTION_DAYS=7 \
#     scripts/maintenance/backup_database.sh
#   scripts/maintenance/backup_database.sh --dry-run

set -euo pipefail

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.prod.yml}"
POSTGRES_SERVICE="${POSTGRES_SERVICE:-postgres}"
POSTGRES_USER="${POSTGRES_USER:-content_user}"
POSTGRES_DB="${POSTGRES_DB:-content_automation}"
BACKUP_DIR="${BACKUP_DIR:-./backups}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
DRY_RUN=false

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: $0 [--dry-run]"
    exit 0
elif [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
backup_file="${BACKUP_DIR%/}/${POSTGRES_DB}_${timestamp}.dump"

if [[ "$DRY_RUN" == "true" ]]; then
    echo "Would create host backup: $backup_file"
    echo "Would execute pg_dump in Compose service: $POSTGRES_SERVICE"
    echo "Would retain backups for $RETENTION_DAYS day(s)"
    exit 0
fi

mkdir -p "$BACKUP_DIR"
chmod 700 "$BACKUP_DIR"

docker compose -f "$COMPOSE_FILE" exec -T "$POSTGRES_SERVICE" \
    pg_dump --format=custom --no-owner --no-acl \
    --username="$POSTGRES_USER" --dbname="$POSTGRES_DB" > "$backup_file"

if [[ ! -s "$backup_file" ]]; then
    echo "Backup failed: output file is empty" >&2
    exit 1
fi

chmod 600 "$backup_file"
find "$BACKUP_DIR" -type f -name "${POSTGRES_DB}_*.dump" \
    -mtime "+${RETENTION_DAYS}" -delete

echo "Backup created: $backup_file"
echo "Copy this file to encrypted off-server storage."
