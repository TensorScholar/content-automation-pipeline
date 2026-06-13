#!/usr/bin/env bash
#
# Restore a custom-format PostgreSQL backup into the production Compose DB.
# This is destructive and requires an explicit --confirm flag.

set -euo pipefail

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.prod.yml}"
POSTGRES_SERVICE="${POSTGRES_SERVICE:-postgres}"
POSTGRES_USER="${POSTGRES_USER:-content_user}"
POSTGRES_DB="${POSTGRES_DB:-content_automation}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: $0 BACKUP_FILE [--confirm]"
    echo "Without --confirm, validates that the backup file is readable."
    exit 0
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 BACKUP_FILE [--confirm]" >&2
    exit 2
fi

backup_file="$1"
confirm="${2:-}"

if [[ ! -r "$backup_file" ]]; then
    echo "Backup file is not readable: $backup_file" >&2
    exit 1
fi

if [[ "$confirm" != "--confirm" ]]; then
    echo "Restore validation passed for: $backup_file"
    echo "Re-run with --confirm to stop app services and replace database contents."
    exit 0
fi

echo "Stopping application services before restore..."
docker compose -f "$COMPOSE_FILE" stop api worker celery-beat frontend nginx

echo "Restoring database from: $backup_file"
docker compose -f "$COMPOSE_FILE" exec -T "$POSTGRES_SERVICE" \
    pg_restore --clean --if-exists --no-owner --no-acl \
    --username="$POSTGRES_USER" --dbname="$POSTGRES_DB" < "$backup_file"

echo "Restore completed. Run the migration and smoke-check commands before restarting traffic."
