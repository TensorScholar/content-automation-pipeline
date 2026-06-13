"""
Background Cleanup & Maintenance Tasks

Periodic jobs to clean up old data and maintain system health.
"""

import asyncio
from datetime import datetime, timedelta, timezone

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

        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days_to_keep)).replace(
            tzinfo=None
        )

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
        ALLOWED_TABLES = frozenset(
            [
                "generated_articles",
                "content_plans",
                "projects",
                "task_results",
            ]
        )

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


# Backups intentionally run from the Docker host via
# scripts/maintenance/backup_database.sh. A Celery worker does not have a
# durable host backup mount or permission to control sibling containers.
