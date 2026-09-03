#!/usr/bin/env bash
# Verify a PostgreSQL backup by restoring it into a disposable database only.
# Production data is never dropped, overwritten, or stopped by this script.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.prod.yml}"
POSTGRES_SERVICE="${POSTGRES_SERVICE:-postgres}"
MIGRATE_SERVICE="${MIGRATE_SERVICE:-migrate}"
POSTGRES_USER="${POSTGRES_USER:-content_user}"
POSTGRES_DB="${POSTGRES_DB:-content_automation}"
EXPECTED_ALEMBIC_HEAD="${EXPECTED_ALEMBIC_HEAD:-}"
CONFIRM=false
BACKUP_FILE=""
TEMP_BACKUP=""
RESTORE_DB="smarlux_restore_verify_$(date -u +%Y%m%d%H%M%S)_$$"

usage() {
  cat <<'USAGE'
Usage:
  scripts/maintenance/verify_backup_restore.sh [BACKUP_FILE]
  scripts/maintenance/verify_backup_restore.sh [BACKUP_FILE] --confirm-disposable-restore

Without --confirm-disposable-restore, this performs non-destructive prerequisites
and backup-format validation only. With confirmation, the backup is restored into
a uniquely named disposable database, verified, and dropped automatically.
If BACKUP_FILE is omitted in confirmed mode, a temporary backup of the current
production database is created and tested.

EXPECTED_ALEMBIC_HEAD may be supplied to enforce a specific release head. For a
temporary backup the source database head is captured automatically. For an
external backup, the repository's single Alembic head is resolved automatically.
USAGE
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

psql_scalar() {
  local database="$1"
  local sql="$2"
  docker compose -f "$COMPOSE_FILE" exec -T "$POSTGRES_SERVICE" \
    psql --username="$POSTGRES_USER" --dbname="$database" --tuples-only --no-align \
    --command="$sql"
}

read_single_alembic_head() {
  local database="$1"
  local head
  head="$(psql_scalar "$database" "
    SELECT CASE
      WHEN COUNT(*) = 1 THEN MIN(version_num)
      ELSE '__INVALID_HEAD_COUNT__:' || COUNT(*)::text
    END
    FROM alembic_version;
  ")"

  if [[ -z "$head" || "$head" == __INVALID_HEAD_COUNT__:* ]]; then
    echo "Restore verification failed: database '$database' does not have exactly one Alembic head (result='$head')" >&2
    return 1
  fi
  printf '%s\n' "$head"
}

resolve_repository_alembic_head() {
  local output
  local -a heads=()

  if ! output="$(
    docker compose -f "$COMPOSE_FILE" --profile migrate run --rm --no-deps "$MIGRATE_SERVICE" \
      alembic heads
  )"; then
    echo "Restore verification failed: unable to resolve repository Alembic head" >&2
    return 1
  fi

  mapfile -t heads < <(
    printf '%s\n' "$output" |
      sed -nE 's/^[[:space:]]*([[:alnum:]_]+)[[:space:]]+\(head\)[[:space:]]*$/\1/p'
  )

  if [[ "${#heads[@]}" -ne 1 ]]; then
    echo "Restore verification failed: expected exactly one repository Alembic head, found ${#heads[@]}" >&2
    printf '%s\n' "$output" >&2
    return 1
  fi
  printf '%s\n' "${heads[0]}"
}

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

  source_head="$(read_single_alembic_head "$POSTGRES_DB")"
  if [[ -n "$EXPECTED_ALEMBIC_HEAD" && "$source_head" != "$EXPECTED_ALEMBIC_HEAD" ]]; then
    echo "Backup verification refused: source Alembic head '$source_head' != expected '$EXPECTED_ALEMBIC_HEAD'" >&2
    exit 1
  fi
  EXPECTED_ALEMBIC_HEAD="${EXPECTED_ALEMBIC_HEAD:-$source_head}"

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

if [[ -z "$EXPECTED_ALEMBIC_HEAD" ]]; then
  EXPECTED_ALEMBIC_HEAD="$(resolve_repository_alembic_head)"
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
  article_revisions
  publishing_attempts
  search_console_connections
  search_console_sync_runs
)
for table in "${required_tables[@]}"; do
  exists="$(psql_scalar "$RESTORE_DB" "SELECT to_regclass('public.${table}') IS NOT NULL;")"
  [[ "$exists" == "t" ]] || { echo "Restore verification failed: missing table $table" >&2; exit 1; }
done

echo "RESTORE_REQUIRED_TABLES_PASS"

head="$(read_single_alembic_head "$RESTORE_DB")"
[[ "$head" == "$EXPECTED_ALEMBIC_HEAD" ]] \
  || { echo "Restore verification failed: Alembic head '$head' != '$EXPECTED_ALEMBIC_HEAD'" >&2; exit 1; }

required_triggers=(
  "article_revisions:trg_article_revisions_assign_identity"
  "article_revisions:trg_article_revisions_prevent_update"
  "generated_articles:trg_generated_articles_capture_revision"
  "generated_articles:trg_generated_articles_validate_current_revision"
)
for spec in "${required_triggers[@]}"; do
  table="${spec%%:*}"
  trigger="${spec#*:}"
  exists="$(psql_scalar "$RESTORE_DB" "
    SELECT EXISTS (
      SELECT 1
      FROM pg_trigger
      WHERE tgrelid = 'public.${table}'::regclass
        AND tgname = '${trigger}'
        AND NOT tgisinternal
        AND tgenabled <> 'D'
    );
  ")"
  [[ "$exists" == "t" ]] \
    || { echo "Restore verification failed: missing or disabled trigger ${trigger} on ${table}" >&2; exit 1; }
done

required_constraints=(
  "generated_articles:fk_generated_articles_current_revision:f"
  "article_revisions:ck_article_revisions_revision_number_positive:c"
  "article_revisions:ck_article_revisions_snapshot_completeness:c"
)
for spec in "${required_constraints[@]}"; do
  table="${spec%%:*}"
  remainder="${spec#*:}"
  constraint="${remainder%%:*}"
  constraint_type="${remainder##*:}"
  exists="$(psql_scalar "$RESTORE_DB" "
    SELECT EXISTS (
      SELECT 1
      FROM pg_constraint
      WHERE conrelid = 'public.${table}'::regclass
        AND conname = '${constraint}'
        AND contype = '${constraint_type}'
        AND convalidated
    );
  ")"
  [[ "$exists" == "t" ]] \
    || { echo "Restore verification failed: missing or unvalidated constraint ${constraint} on ${table}" >&2; exit 1; }
done

revision_index="$(psql_scalar "$RESTORE_DB" "SELECT to_regclass('public.uq_article_revisions_article_number') IS NOT NULL;")"
[[ "$revision_index" == "t" ]] \
  || { echo "Restore verification failed: missing unique revision-number index" >&2; exit 1; }

echo "RESTORE_REVISION_SCHEMA_GUARDS_PASS"

integrity="$(psql_scalar "$RESTORE_DB" "
  SELECT
    NOT EXISTS (
      SELECT 1
      FROM generated_articles AS article
      LEFT JOIN article_revisions AS revision
        ON revision.id = article.current_revision_id
       AND revision.article_id = article.id
      WHERE article.current_revision_id IS NOT NULL
        AND revision.id IS NULL
    )
    AND NOT EXISTS (
      SELECT 1
      FROM article_revisions
      WHERE revision_number IS NULL OR revision_number <= 0
    );
")"
[[ "$integrity" == "t" ]] \
  || { echo "Restore verification failed: immutable revision ledger integrity check failed" >&2; exit 1; }

echo "RESTORE_REVISION_INTEGRITY_PASS"
echo "DISPOSABLE_RESTORE_PASS database=$RESTORE_DB alembic_head=$head"
echo "The disposable database will now be dropped automatically."
