"""
Background Cleanup & Maintenance Tasks

Periodic jobs to clean up old data, maintain system health,
and run automated database backups.
"""

import asyncio
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from celery import shared_task
from loguru import logger
from sqlalchemy.sql import text

from api.dependencies import get_database
from infrastructure.redis_client import RedisClient


@shared_task(name="cleanup_old_task_results")
def cleanup_old_task_results_task():
    """Celery task to archive old task results (runs daily)."""
    asyncio.run(cleanup_old_task_results())


async def cleanup_old_task_results(days_to_keep: int = 30):
    """
    Archive or delete task results older than specified days.

    Keeps the database lean and prevents unbounded growth.
    """
    try:
        db = get_database()
        await db.initialize()

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)

        async with db.get_writer_session() as session:
            # Count old tasks
            count_result = await session.execute(
                text("""
                    SELECT COUNT(*) FROM task_results
                    WHERE created_at < :cutoff_date
                    AND status IN ('SUCCESS', 'FAILURE')
                """),
                {"cutoff_date": cutoff_date},
            )
            old_count = count_result.scalar()

            if old_count == 0:
                logger.info("No old task results to clean up")
                return

            # Delete old completed/failed tasks
            result = await session.execute(
                text("""
                    DELETE FROM task_results
                    WHERE created_at < :cutoff_date
                    AND status IN ('SUCCESS', 'FAILURE')
                """),
                {"cutoff_date": cutoff_date},
            )

            await session.commit()

            deleted_count = result.rowcount
            logger.info(
                f"Cleaned up {deleted_count} old task results " f"(older than {days_to_keep} days)"
            )

            return {
                "deleted_count": deleted_count,
                "cutoff_date": cutoff_date.isoformat(),
            }

    except Exception as e:
        logger.error(f"Failed to cleanup old task results: {e}")
        raise
    finally:
        await db.close()


@shared_task(name="cleanup_expired_cache")
def cleanup_expired_cache_task():
    """Celery task to clean up expired Redis cache entries."""
    asyncio.run(cleanup_expired_cache())


async def cleanup_expired_cache():
    """
    Clean up expired cache entries from Redis.

    Redis automatically removes expired keys, but this helps with memory optimization.
    """
    try:
        from api.dependencies import get_redis

        redis = get_redis()
        await redis.initialize()

        # Get cache statistics
        info = await redis.info()

        logger.info(
            f"Redis cache status: "
            f"used_memory={info.get('used_memory_human', 'N/A')}, "
            f"keys={info.get('db0', {}).get('keys', 0)}"
        )

        # Optionally: Clean up specific cache patterns that are expired
        # This is mostly handled automatically by Redis TTL

        return {
            "status": "ok",
            "memory_used": info.get("used_memory_human"),
            "total_keys": info.get("db0", {}).get("keys", 0),
        }

    except Exception as e:
        logger.error(f"Failed to cleanup expired cache: {e}")
        raise


@shared_task(name="vacuum_analyze_database")
def vacuum_analyze_database_task():
    """Celery task to run VACUUM ANALYZE on database (weekly)."""
    asyncio.run(vacuum_analyze_database())


async def vacuum_analyze_database():
    """
    Run VACUUM ANALYZE to optimize database performance.

    This reclaims storage and updates query planner statistics.
    """
    try:
        db = get_database()
        await db.initialize()

        # FIX: Use whitelist to prevent SQL injection
        # Table names cannot be parameterized, so we use a strict whitelist
        ALLOWED_TABLES = frozenset([
            "generated_articles",
            "content_plans",
            "projects",
            "task_results",
        ])

        optimized_count = 0
        async with db.get_writer_session() as session:
            for table in ALLOWED_TABLES:
                try:
                    # VACUUM ANALYZE can't run in a transaction
                    # Table name is safe (from hardcoded whitelist)
                    await session.execute(text(f"VACUUM ANALYZE {table}"))
                    logger.info(f"VACUUM ANALYZE completed for {table}")
                    optimized_count += 1
                except Exception as e:
                    logger.warning(f"VACUUM ANALYZE failed for {table}: {e}")

        logger.info(f"Database optimization completed: {optimized_count} tables")
        return {"status": "ok", "tables_optimized": optimized_count}

    except Exception as e:
        logger.error(f"Failed to vacuum analyze database: {e}")
        raise
    finally:
        await db.close()


# =============================================================================
# DATABASE BACKUP TASK
# =============================================================================


@shared_task(name="backup_database", bind=True, max_retries=2, default_retry_delay=300)
def backup_database_task(self):
    """
    Celery task to run automated PostgreSQL backup.

    Runs daily via Beat scheduler. Uses pg_dump either through Docker exec
    or directly, compresses with gzip, and rotates old backups.

    Backup location: /var/backups/postgres (configurable via BACKUP_DIR env).
    Retention: 7 days by default (configurable via RETENTION_DAYS env).
    """
    try:
        result = run_database_backup()
        logger.info(f"Database backup completed: {result}")
        return result
    except Exception as e:
        logger.error(f"Database backup failed: {e}")
        raise self.retry(exc=e)


def run_database_backup() -> dict:
    """
    Execute database backup with compression and rotation.

    Returns:
        dict with backup status, file path, size, and cleanup info.
    """
    backup_dir = Path(os.getenv("BACKUP_DIR", "/var/backups/postgres"))
    retention_days = int(os.getenv("RETENTION_DAYS", "7"))
    postgres_container = os.getenv("POSTGRES_CONTAINER", "content-automation-postgres-prod")
    postgres_user = os.getenv("POSTGRES_USER", "content_user")
    postgres_db = os.getenv("POSTGRES_DB", "content_automation")
    database_url = os.getenv("DATABASE_URL", "")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"backup_{postgres_db}_{timestamp}.sql.gz"

    # Ensure backup directory exists
    backup_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting database backup → {backup_file}")

    # Strategy 1: If running inside Docker alongside postgres, use pg_dump directly
    # Strategy 2: If running on host, use docker exec
    # Strategy 3: If DATABASE_URL is available, use pg_dump with connection string
    backup_success = False
    backup_method = "unknown"

    # Try direct pg_dump first (works inside Docker or when postgres tools are installed)
    if database_url:
        try:
            # Parse sync URL for pg_dump (strip asyncpg driver prefix)
            dump_url = database_url.replace("postgresql+asyncpg://", "postgresql://")
            cmd = f'pg_dump "{dump_url}" | gzip > "{backup_file}"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
            if result.returncode == 0 and backup_file.exists() and backup_file.stat().st_size > 0:
                backup_success = True
                backup_method = "pg_dump_direct"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.debug("Direct pg_dump not available, trying docker exec")

    # Fallback: docker exec
    if not backup_success:
        try:
            cmd = (
                f"docker exec {postgres_container} "
                f"pg_dump -U {postgres_user} {postgres_db} "
                f"| gzip > {backup_file}"
            )
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
            if result.returncode == 0 and backup_file.exists() and backup_file.stat().st_size > 0:
                backup_success = True
                backup_method = "docker_exec"
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Docker exec backup failed: {e}")

    if not backup_success:
        # Clean up empty/partial file
        if backup_file.exists():
            backup_file.unlink()
        raise RuntimeError(
            f"Database backup failed. Tried pg_dump and docker exec. "
            f"Ensure pg_dump is available or container '{postgres_container}' is running."
        )

    backup_size = backup_file.stat().st_size
    backup_size_mb = backup_size / (1024 * 1024)
    logger.info(
        f"Backup completed: {backup_file.name} ({backup_size_mb:.2f} MB) via {backup_method}"
    )

    # Rotate old backups
    deleted_count = 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    for old_file in backup_dir.glob("backup_*.sql.gz"):
        try:
            file_mtime = datetime.fromtimestamp(old_file.stat().st_mtime, tz=timezone.utc)
            if file_mtime < cutoff:
                old_file.unlink()
                deleted_count += 1
                logger.debug(f"Deleted old backup: {old_file.name}")
        except Exception as e:
            logger.warning(f"Failed to delete old backup {old_file.name}: {e}")

    if deleted_count > 0:
        logger.info(f"Rotated {deleted_count} backup(s) older than {retention_days} days")

    return {
        "status": "success",
        "file": str(backup_file),
        "size_mb": round(backup_size_mb, 2),
        "method": backup_method,
        "rotated_count": deleted_count,
        "timestamp": timestamp,
    }
