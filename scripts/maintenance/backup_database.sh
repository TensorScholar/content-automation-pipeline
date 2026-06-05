#!/bin/bash
# =============================================================================
# PostgreSQL Automated Backup Script
# =============================================================================
#
# Features:
# - Compressed backups with gzip
# - 7-day retention with automatic rotation
# - Optional S3 upload
# - Docker and standalone PostgreSQL support
# - Exit codes for monitoring integration
#
# Usage:
#   ./backup_database.sh                    # Full backup
#   ./backup_database.sh --dry-run          # Test without executing
#   ./backup_database.sh --upload-s3        # Backup and upload to S3
#
# Cron example (daily at 2 AM):
#   0 2 * * * /path/to/backup_database.sh >> /var/log/db-backup.log 2>&1
# =============================================================================

set -euo pipefail

# Configuration (override via environment variables)
BACKUP_DIR="${BACKUP_DIR:-/var/backups/postgres}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
POSTGRES_CONTAINER="${POSTGRES_CONTAINER:-content-automation-postgres-prod}"
POSTGRES_USER="${POSTGRES_USER:-content_user}"
POSTGRES_DB="${POSTGRES_DB:-content_automation}"
S3_BUCKET="${S3_BUCKET:-}"
S3_PREFIX="${S3_PREFIX:-backups/postgres}"
DRY_RUN=false
UPLOAD_S3=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --upload-s3)
            UPLOAD_S3=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Timestamp for backup file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/backup_${POSTGRES_DB}_${TIMESTAMP}.sql.gz"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Create backup directory if not exists
create_backup_dir() {
    if [[ ! -d "$BACKUP_DIR" ]]; then
        log "Creating backup directory: $BACKUP_DIR"
        if [[ "$DRY_RUN" == "false" ]]; then
            mkdir -p "$BACKUP_DIR"
            chmod 700 "$BACKUP_DIR"
        fi
    fi
}

# Perform database backup
perform_backup() {
    log "Starting backup of database: $POSTGRES_DB"
    log "Backup file: $BACKUP_FILE"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would execute: docker exec $POSTGRES_CONTAINER pg_dump -U $POSTGRES_USER $POSTGRES_DB | gzip > $BACKUP_FILE"
        return 0
    fi

    # Check if running in Docker environment
    if docker ps --format '{{.Names}}' | grep -q "^${POSTGRES_CONTAINER}$"; then
        log "Using Docker container: $POSTGRES_CONTAINER"
        docker exec "$POSTGRES_CONTAINER" pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB" | gzip > "$BACKUP_FILE"
    else
        log "Using local pg_dump"
        pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB" | gzip > "$BACKUP_FILE"
    fi

    # Verify backup was created
    if [[ -f "$BACKUP_FILE" ]]; then
        BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
        log "Backup completed successfully: $BACKUP_SIZE"
    else
        log "ERROR: Backup file was not created"
        exit 1
    fi
}

# Upload to S3 (optional)
upload_to_s3() {
    if [[ "$UPLOAD_S3" == "true" && -n "$S3_BUCKET" ]]; then
        S3_PATH="s3://${S3_BUCKET}/${S3_PREFIX}/$(basename "$BACKUP_FILE")"
        log "Uploading to S3: $S3_PATH"

        if [[ "$DRY_RUN" == "true" ]]; then
            log "[DRY RUN] Would execute: aws s3 cp $BACKUP_FILE $S3_PATH"
            return 0
        fi

        aws s3 cp "$BACKUP_FILE" "$S3_PATH" --only-show-errors
        log "S3 upload completed"
    elif [[ "$UPLOAD_S3" == "true" && -z "$S3_BUCKET" ]]; then
        log "WARNING: --upload-s3 specified but S3_BUCKET not set"
    fi
}

# Cleanup old backups
cleanup_old_backups() {
    log "Cleaning up backups older than $RETENTION_DAYS days"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would delete:"
        find "$BACKUP_DIR" -name "backup_*.sql.gz" -type f -mtime +"$RETENTION_DAYS" -print
        return 0
    fi

    DELETED_COUNT=$(find "$BACKUP_DIR" -name "backup_*.sql.gz" -type f -mtime +"$RETENTION_DAYS" -delete -print | wc -l)
    log "Deleted $DELETED_COUNT old backup(s)"
}

# Main execution
main() {
    log "=========================================="
    log "PostgreSQL Backup Script"
    log "=========================================="

    if [[ "$DRY_RUN" == "true" ]]; then
        log "Running in DRY RUN mode - no changes will be made"
    fi

    create_backup_dir
    perform_backup
    upload_to_s3
    cleanup_old_backups

    log "Backup process completed successfully"
    log "=========================================="
}

main
